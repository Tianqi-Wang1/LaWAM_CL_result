#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence


VARIANT_SPECS = {
    "taskwm_hdhz": ("task", "hdhz"),
    "taskwm_hdh": ("task", "hdh"),
    "taskwm_dh": ("task", "dh"),
    "basewm_hdhz": ("base", "hdhz"),
    "basewm_hdh": ("base", "hdh"),
    "basewm_dh": ("base", "dh"),
}
VARIANTS = tuple(VARIANT_SPECS)
EPS = 1e-12


def read_rows(root: Path):
    paths = sorted(root.glob("T*/dynamics_2x3_probe.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No T*/dynamics_2x3_probe.jsonl under {root}")
    rows = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_source_file"] = str(p)
                    rows.append(r)
    if not rows:
        raise RuntimeError("No 2x3 probe rows")
    return rows, paths


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def pair_share(a: float, b: float) -> tuple[float, float]:
    den = abs(a) + abs(b) + EPS
    return a / den, b / den


def adaptive_lambda(conf: float, delta: float, lambda_max: float, gamma: float) -> float:
    if delta <= 0.0 or conf >= delta:
        return 0.0
    x = max(0.0, min(1.0, (delta - conf) / delta))
    return max(0.0, min(1.0, lambda_max * (x ** gamma)))


def fused_winner(r: dict, variant: str, lam: float) -> int:
    top2 = [int(x) for x in r["semantic_top2_tasks"]]
    if len(top2) == 1:
        return top2[0]
    if len(top2) != 2:
        raise ValueError(f"Expected Top-2 pair, got {top2}")
    a, b = top2
    sem = r["semantic_errors"]
    dyn = r["variants"][variant]["dynamics_errors_top2"]
    esa, esb = float(sem[str(a)]), float(sem[str(b)])
    eda, edb = float(dyn[str(a)]), float(dyn[str(b)])
    ssa, ssb = pair_share(esa, esb)
    dsa, dsb = pair_share(eda, edb)
    fa = (1.0 - lam) * ssa + lam * dsa
    fb = (1.0 - lam) * ssb + lam * dsb
    return a if fa <= fb else b


def evaluate_variant(rows: Sequence[dict], variant: str, delta: float, lambda_max: float, gamma: float):
    n = len(rows)
    sem_ok_n = 0
    dyn_ok_n = 0
    hard_rec = hard_dmg = 0
    fusion_ok_n = fusion_rec = fusion_dmg = 0
    activated = 0
    lambda_sum = 0.0
    lambda_active_sum = 0.0
    sem_err_gated = 0

    for r in rows:
        gt = int(r["gt_task_id"])
        sem = int(r["semantic_top1_task"])
        sem_ok = sem == gt
        sem_ok_n += int(sem_ok)
        vr = r["variants"][variant]
        dyn = int(vr["dynamics_winner_task"])
        dyn_ok = dyn == gt
        dyn_ok_n += int(dyn_ok)
        hard_rec += int((not sem_ok) and dyn_ok)
        hard_dmg += int(sem_ok and (not dyn_ok))

        conf = float(r["semantic_normalized_gap"])
        gate = conf < delta and len(r["semantic_top2_tasks"]) >= 2
        lam = adaptive_lambda(conf, delta, lambda_max, gamma) if gate else 0.0
        if gate:
            activated += 1
            sem_err_gated += int(not sem_ok)
            lambda_active_sum += lam
        lambda_sum += lam
        fw = fused_winner(r, variant, lam)
        fok = fw == gt
        fusion_ok_n += int(fok)
        fusion_rec += int((not sem_ok) and fok)
        fusion_dmg += int(sem_ok and (not fok))

    sem_err = n - sem_ok_n
    return {
        "variant": variant,
        "wm_source": VARIANT_SPECS[variant][0],
        "input_mode": VARIANT_SPECS[variant][1],
        "n": n,
        "semantic_accuracy": sem_ok_n / n,
        "dynamics_top2_accuracy": dyn_ok_n / n,
        "hard_recovered": hard_rec,
        "hard_recovery_rate": hard_rec / sem_err if sem_err else math.nan,
        "hard_damaged": hard_dmg,
        "hard_damage_rate": hard_dmg / sem_ok_n if sem_ok_n else math.nan,
        "hard_net_corrections": hard_rec - hard_dmg,
        "gate_threshold": delta,
        "gate_activation_count": activated,
        "gate_activation_rate": activated / n,
        "semantic_errors_total": sem_err,
        "semantic_errors_gated": sem_err_gated,
        "semantic_error_capture_rate": sem_err_gated / sem_err if sem_err else 1.0,
        "lambda_max": lambda_max,
        "gamma": gamma,
        "mean_lambda_all": lambda_sum / n,
        "mean_lambda_activated": lambda_active_sum / activated if activated else 0.0,
        "hybrid_accuracy": fusion_ok_n / n,
        "gain_vs_semantic": fusion_ok_n / n - sem_ok_n / n,
        "fusion_recovered": fusion_rec,
        "fusion_recovery_rate": fusion_rec / sem_err if sem_err else math.nan,
        "fusion_damaged": fusion_dmg,
        "fusion_damage_rate": fusion_dmg / sem_ok_n if sem_ok_n else math.nan,
        "fusion_net_corrections": fusion_rec - fusion_dmg,
    }


def percentile_bins(rows: Sequence[dict], n_bins: int = 10):
    ordered = sorted(rows, key=lambda r: float(r["semantic_normalized_gap"]))
    bins = []
    n = len(ordered)
    for b in range(n_bins):
        lo = (b * n) // n_bins
        hi = ((b + 1) * n) // n_bins
        rr = ordered[lo:hi]
        if rr:
            bins.append((b, rr))
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--gate-threshold", type=float, default=0.20)
    ap.add_argument("--lambda-max", type=float, default=0.50)
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "dynamics_2x3_summary"
    )
    out.mkdir(parents=True, exist_ok=True)
    rows, paths = read_rows(root)
    n = len(rows)
    gt_tasks = sorted(set(int(r["gt_task_id"]) for r in rows))
    sem_acc = sum(bool(r["semantic_top1_correct"]) for r in rows) / n
    top2 = sum(bool(r["semantic_top2_correct"]) for r in rows) / n

    summaries = [
        evaluate_variant(
            rows,
            v,
            float(args.gate_threshold),
            float(args.lambda_max),
            float(args.gamma),
        )
        for v in VARIANTS
    ]
    write_csv(out / "variant_summary.csv", summaries)

    # Paired per-task metrics.
    task_rows = []
    for v in VARIANTS:
        for t in gt_tasks:
            rr = [r for r in rows if int(r["gt_task_id"]) == t]
            s = evaluate_variant(
                rr,
                v,
                float(args.gate_threshold),
                float(args.lambda_max),
                float(args.gamma),
            )
            task_rows.append({"gt_task_id": t, **s})
    write_csv(out / "variant_per_task.csv", task_rows)

    # Long-format confusion for hard Dynamics winners.
    conf_rows = []
    for v in VARIANTS:
        conf = Counter()
        for r in rows:
            conf[(int(r["gt_task_id"]), int(r["variants"][v]["dynamics_winner_task"]))] += 1
        for (gt, pred), count in sorted(conf.items()):
            conf_rows.append({"variant": v, "gt_task_id": gt, "pred_task_id": pred, "count": count})
    write_csv(out / "variant_dynamics_confusion_long.csv", conf_rows)

    # Same confidence bins for every variant; ideal for checking whether the
    # verifier helps only in low-confidence regions.
    decile_rows = []
    for b, rr in percentile_bins(rows, 10):
        sem = sum(bool(r["semantic_top1_correct"]) for r in rr) / len(rr)
        for v in VARIANTS:
            dyn = sum(bool(r["variants"][v]["dynamics_correct"]) for r in rr) / len(rr)
            rec = sum(bool(r["variants"][v]["semantic_error_recovered"]) for r in rr)
            dmg = sum(bool(r["variants"][v]["semantic_correct_damaged"]) for r in rr)
            decile_rows.append({
                "bin": b,
                "variant": v,
                "n": len(rr),
                "confidence_min": min(float(r["semantic_normalized_gap"]) for r in rr),
                "confidence_max": max(float(r["semantic_normalized_gap"]) for r in rr),
                "semantic_accuracy": sem,
                "dynamics_accuracy": dyn,
                "dynamics_minus_semantic": dyn - sem,
                "recovered": rec,
                "damaged": dmg,
                "hard_net_corrections": rec - dmg,
            })
    write_csv(out / "variant_confidence_deciles.csv", decile_rows)

    # Compact 2x3 matrices, one for raw Dynamics and one for fixed-hyperparameter hybrid.
    smap = {r["variant"]: r for r in summaries}
    matrix_rows = []
    for wm in ("task", "base"):
        row = {"wm_source": wm}
        for mode in ("hdhz", "hdh", "dh"):
            v = f"{wm}wm_{mode}"
            row[f"dyn_{mode}"] = smap[v]["dynamics_top2_accuracy"]
            row[f"hybrid_{mode}"] = smap[v]["hybrid_accuracy"]
            row[f"net_{mode}"] = smap[v]["fusion_net_corrections"]
        matrix_rows.append(row)
    write_csv(out / "matrix_2x3.csv", matrix_rows)

    # Paired winner table: which variant is best on the exact same chunks.
    best_hybrid = max(
        summaries,
        key=lambda x: (float(x["hybrid_accuracy"]), float(x["fusion_net_corrections"])),
    )
    best_dyn = max(summaries, key=lambda x: float(x["dynamics_top2_accuracy"]))
    summary = {
        "root": str(root),
        "source_logs": [str(p) for p in paths],
        "num_decisions": n,
        "semantic_top1_accuracy": sem_acc,
        "semantic_top2_recall": top2,
        "fixed_fusion_hyperparameters": {
            "gate_threshold": float(args.gate_threshold),
            "lambda_max": float(args.lambda_max),
            "gamma": float(args.gamma),
            "note": "Identical for all six variants; no per-variant retuning.",
        },
        "variants": summaries,
        "best_dynamics_variant": best_dyn["variant"],
        "best_hybrid_variant": best_hybrid["variant"],
        "warning": "This is a passive oracle-action probe. Use it to select representation/WM-source ablations before closed-loop evaluation.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("======================================================================")
    print(" Routing-V2 paired 2x3 passive future-dynamics ablation")
    print(f" Decisions            : {n}")
    print(f" Semantic Top1        : {100*sem_acc:.2f}%")
    print(f" Semantic Top2 recall : {100*top2:.2f}%")
    print(
        f" Fusion hyperparams   : delta={args.gate_threshold:.3f}, "
        f"lambda_max={args.lambda_max:.3f}, gamma={args.gamma:.3f}"
    )
    print("----------------------------------------------------------------------")
    print(" Variant         DynTop2   Recovery Damage   Net   Hybrid   Gain")
    for s in summaries:
        print(
            f" {s['variant']:<15} "
            f"{100*float(s['dynamics_top2_accuracy']):7.2f}% "
            f"{int(s['fusion_recovered']):8d} {int(s['fusion_damaged']):6d} "
            f"{int(s['fusion_net_corrections']):5d} "
            f"{100*float(s['hybrid_accuracy']):7.2f}% "
            f"{100*float(s['gain_vs_semantic']):+6.2f}pp"
        )
    print("----------------------------------------------------------------------")
    print(f" Best raw Dynamics    : {best_dyn['variant']}")
    print(f" Best fixed-fusion    : {best_hybrid['variant']}")
    print(f" Output               : {out}")
    print("======================================================================")


if __name__ == "__main__":
    main()
