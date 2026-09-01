#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

EPS = 1e-12


def parse_float_list(text: str) -> list[float]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    if not vals:
        raise ValueError("Expected at least one float")
    return vals


def read_rows(root: Path) -> tuple[list[dict], list[Path]]:
    paths = sorted(root.glob("T*/dynamics_probe.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No T*/dynamics_probe.jsonl under {root}")
    rows: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_source_file"] = str(p)
                rows.append(r)
    if not rows:
        raise RuntimeError(f"No rows in {root}")
    return rows, paths


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def quantile(vals: Sequence[float], q: float) -> float:
    xs = sorted(float(v) for v in vals)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def pair_share(err_a: float, err_b: float) -> tuple[float, float]:
    """Lower is better. Normalize two non-negative errors to a comparable pair share."""
    den = abs(err_a) + abs(err_b) + EPS
    return err_a / den, err_b / den


def candidate_errors(r: dict) -> tuple[int, int, float, float, float, float]:
    top2 = [int(x) for x in r["semantic_top2_tasks"]]
    if len(top2) != 2:
        raise ValueError(f"Expected exactly two semantic candidates, got {top2}")
    a, b = top2
    sem = r["semantic_errors"]
    dyn = r["dynamics_errors_top2"]
    return (
        a,
        b,
        float(sem[str(a)]),
        float(sem[str(b)]),
        float(dyn[str(a)]),
        float(dyn[str(b)]),
    )


def choose_fused(r: dict, lambda_dyn: float) -> tuple[int, dict]:
    a, b, esa, esb, eda, edb = candidate_errors(r)
    ssa, ssb = pair_share(esa, esb)
    dsa, dsb = pair_share(eda, edb)
    fa = (1.0 - lambda_dyn) * ssa + lambda_dyn * dsa
    fb = (1.0 - lambda_dyn) * ssb + lambda_dyn * dsb
    winner = a if fa <= fb else b
    return winner, {
        "candidate_a": a,
        "candidate_b": b,
        "sem_share_a": ssa,
        "sem_share_b": ssb,
        "dyn_share_a": dsa,
        "dyn_share_b": dsb,
        "fused_a": fa,
        "fused_b": fb,
    }


def adaptive_lambda(conf: float, delta: float, lambda_max: float, gamma: float) -> float:
    if delta <= 0.0 or conf >= delta:
        return 0.0
    x = max(0.0, min(1.0, (delta - conf) / delta))
    return max(0.0, min(1.0, lambda_max * (x ** gamma)))


def evaluate_variant(
    rows: Sequence[dict],
    *,
    mode: str,
    gate_threshold: float,
    lambda_value: float,
    gamma: float = 1.0,
) -> tuple[dict, list[dict]]:
    decisions: list[dict] = []
    sem_correct_n = 0
    hybrid_correct_n = 0
    recovered = 0
    damaged = 0
    activated = 0
    sem_errors_gated = 0
    lambda_sum = 0.0
    lambda_activated_sum = 0.0

    for idx, r in enumerate(rows):
        gt = int(r["gt_task_id"])
        sem = int(r["semantic_top1_task"])
        conf = float(r["semantic_normalized_gap"])
        sem_ok = sem == gt
        sem_correct_n += int(sem_ok)
        use_gate = conf < gate_threshold

        if mode == "fixed":
            lam = lambda_value if use_gate else 0.0
        elif mode == "adaptive":
            lam = adaptive_lambda(conf, gate_threshold, lambda_value, gamma)
        elif mode == "hard_dyn":
            lam = 1.0 if use_gate else 0.0
        elif mode == "semantic":
            lam = 0.0
            use_gate = False
        else:
            raise ValueError(mode)

        if use_gate:
            activated += 1
            sem_errors_gated += int(not sem_ok)

        winner, detail = choose_fused(r, lam)
        ok = winner == gt
        hybrid_correct_n += int(ok)
        recovered += int((not sem_ok) and ok)
        damaged += int(sem_ok and (not ok))
        lambda_sum += lam
        if use_gate:
            lambda_activated_sum += lam

        decisions.append({
            "row_index": idx,
            "decision_id": r.get("decision_id", idx),
            "gt_task_id": gt,
            "semantic_top1_task": sem,
            "semantic_correct": sem_ok,
            "semantic_confidence": conf,
            "gate_active": use_gate,
            "lambda_dyn": lam,
            "fusion_winner_task": winner,
            "fusion_correct": ok,
            "recovered": (not sem_ok) and ok,
            "damaged": sem_ok and (not ok),
            **detail,
        })

    n = len(rows)
    sem_errors_total = n - sem_correct_n
    sem_ok_total = sem_correct_n
    hybrid_acc = hybrid_correct_n / n
    sem_acc = sem_correct_n / n
    summary = {
        "mode": mode,
        "gate_threshold": gate_threshold,
        "lambda_value": lambda_value,
        "gamma": gamma,
        "n": n,
        "activation_count": activated,
        "activation_rate": activated / n,
        "mean_lambda_all": lambda_sum / n,
        "mean_lambda_activated": lambda_activated_sum / activated if activated else 0.0,
        "semantic_accuracy": sem_acc,
        "hybrid_accuracy": hybrid_acc,
        "gain_vs_semantic": hybrid_acc - sem_acc,
        "semantic_errors_total": sem_errors_total,
        "semantic_errors_gated": sem_errors_gated,
        "semantic_error_capture_rate": sem_errors_gated / sem_errors_total if sem_errors_total else 1.0,
        "recovered": recovered,
        "recovery_rate_all_semantic_errors": recovered / sem_errors_total if sem_errors_total else 1.0,
        "damaged": damaged,
        "damage_rate_all_semantic_correct": damaged / sem_ok_total if sem_ok_total else 0.0,
        "net_corrections": recovered - damaged,
    }
    return summary, decisions


def per_task(rows: Sequence[dict], decisions: Sequence[dict], variant_name: str) -> list[dict]:
    by_t_rows: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for r, d in zip(rows, decisions):
        by_t_rows[int(r["gt_task_id"])].append((r, d))
    out = []
    for t in sorted(by_t_rows):
        pairs = by_t_rows[t]
        n = len(pairs)
        sem_ok = sum(bool(r["semantic_top1_correct"]) for r, _ in pairs)
        fusion_ok = sum(bool(d["fusion_correct"]) for _, d in pairs)
        rec = sum(bool(d["recovered"]) for _, d in pairs)
        dmg = sum(bool(d["damaged"]) for _, d in pairs)
        sem_err = n - sem_ok
        out.append({
            "variant": variant_name,
            "gt_task_id": t,
            "n": n,
            "semantic_accuracy": sem_ok / n,
            "fusion_accuracy": fusion_ok / n,
            "gain_vs_semantic": (fusion_ok - sem_ok) / n,
            "semantic_errors": sem_err,
            "recovered": rec,
            "recovery_rate": rec / sem_err if sem_err else math.nan,
            "semantic_correct": sem_ok,
            "damaged": dmg,
            "damage_rate": dmg / sem_ok if sem_ok else math.nan,
            "net_corrections": rec - dmg,
        })
    return out


def confidence_deciles(rows: Sequence[dict], decisions: Sequence[dict], variant_name: str) -> list[dict]:
    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["semantic_normalized_gap"]))
    n = len(order)
    out = []
    for b in range(10):
        lo = int(round(b * n / 10))
        hi = int(round((b + 1) * n / 10))
        ids = order[lo:hi]
        if not ids:
            continue
        sem_ok = sum(bool(rows[i]["semantic_top1_correct"]) for i in ids)
        fus_ok = sum(bool(decisions[i]["fusion_correct"]) for i in ids)
        rec = sum(bool(decisions[i]["recovered"]) for i in ids)
        dmg = sum(bool(decisions[i]["damaged"]) for i in ids)
        active = sum(bool(decisions[i]["gate_active"]) for i in ids)
        mean_conf = sum(float(rows[i]["semantic_normalized_gap"]) for i in ids) / len(ids)
        mean_lam = sum(float(decisions[i]["lambda_dyn"]) for i in ids) / len(ids)
        out.append({
            "variant": variant_name,
            "bin": b,
            "n": len(ids),
            "mean_confidence": mean_conf,
            "activation_rate": active / len(ids),
            "mean_lambda_dyn": mean_lam,
            "semantic_accuracy": sem_ok / len(ids),
            "fusion_accuracy": fus_ok / len(ids),
            "fusion_minus_semantic": (fus_ok - sem_ok) / len(ids),
            "recovered": rec,
            "damaged": dmg,
            "net_corrections": rec - dmg,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline confidence-gated Semantic+Dynamics fusion analysis for Routing-V2")
    ap.add_argument("--root", required=True, help="Dynamics passive-probe root containing T*/dynamics_probe.jsonl")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--primary-gate", type=float, default=0.20, help="Primary semantic confidence threshold; Dynamics considered only below this value")
    ap.add_argument("--gate-thresholds", default="0.03,0.05,0.10,0.15,0.20,0.25", help="Diagnostic threshold sweep")
    ap.add_argument("--lambdas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--lambda-maxes", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--gammas", default="0.5,1,2")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "confidence_fusion"
    out.mkdir(parents=True, exist_ok=True)
    rows, paths = read_rows(root)

    lambdas = parse_float_list(args.lambdas)
    lambda_maxes = parse_float_list(args.lambda_maxes)
    gates = parse_float_list(args.gate_thresholds)
    if args.primary_gate not in gates:
        gates = sorted(set(gates + [args.primary_gate]))
    gammas = parse_float_list(args.gammas)

    sweep_rows: list[dict] = []
    variants: dict[str, tuple[dict, list[dict]]] = {}

    sem_s, sem_d = evaluate_variant(rows, mode="semantic", gate_threshold=args.primary_gate, lambda_value=0.0)
    variants["semantic_only"] = (sem_s, sem_d)

    hard_s, hard_d = evaluate_variant(rows, mode="hard_dyn", gate_threshold=args.primary_gate, lambda_value=1.0)
    variants[f"hard_dyn_gate_d{args.primary_gate:g}"] = (hard_s, hard_d)

    for delta in gates:
        for lam in lambdas:
            s, d = evaluate_variant(rows, mode="fixed", gate_threshold=delta, lambda_value=lam)
            sweep_rows.append({"family": "fixed", **s})
        for gamma in gammas:
            for lm in lambda_maxes:
                s, d = evaluate_variant(rows, mode="adaptive", gate_threshold=delta, lambda_value=lm, gamma=gamma)
                sweep_rows.append({"family": "adaptive", **s})

    write_csv(out / "fusion_sweep.csv", sweep_rows)

    # Diagnostic best by hybrid accuracy, tie-break lower activation then lower mean lambda.
    fixed_primary = [r for r in sweep_rows if r["family"] == "fixed" and abs(float(r["gate_threshold"]) - args.primary_gate) < 1e-12]
    adaptive_primary = [r for r in sweep_rows if r["family"] == "adaptive" and abs(float(r["gate_threshold"]) - args.primary_gate) < 1e-12]
    best_fixed_primary = max(fixed_primary, key=lambda r: (float(r["hybrid_accuracy"]), -float(r["mean_lambda_all"])))
    best_adaptive_primary = max(adaptive_primary, key=lambda r: (float(r["hybrid_accuracy"]), -float(r["mean_lambda_all"])))
    best_global = max(sweep_rows, key=lambda r: (float(r["hybrid_accuracy"]), -float(r["activation_rate"]), -float(r["mean_lambda_all"])))

    # Re-evaluate selected variants to retain per-decision details.
    bf_s, bf_d = evaluate_variant(
        rows,
        mode="fixed",
        gate_threshold=float(best_fixed_primary["gate_threshold"]),
        lambda_value=float(best_fixed_primary["lambda_value"]),
    )
    ba_s, ba_d = evaluate_variant(
        rows,
        mode="adaptive",
        gate_threshold=float(best_adaptive_primary["gate_threshold"]),
        lambda_value=float(best_adaptive_primary["lambda_value"]),
        gamma=float(best_adaptive_primary["gamma"]),
    )
    bg_s, bg_d = evaluate_variant(
        rows,
        mode=str(best_global["family"]),
        gate_threshold=float(best_global["gate_threshold"]),
        lambda_value=float(best_global["lambda_value"]),
        gamma=float(best_global["gamma"]),
    )
    variants[f"fixed_primary_best"] = (bf_s, bf_d)
    variants[f"adaptive_primary_best"] = (ba_s, ba_d)
    variants[f"global_diagnostic_best"] = (bg_s, bg_d)

    # Summaries for selected variants.
    selected_rows = []
    pt_rows: list[dict] = []
    decile_rows: list[dict] = []
    for name, (s, d) in variants.items():
        selected_rows.append({"variant": name, **s})
        pt_rows.extend(per_task(rows, d, name))
        decile_rows.extend(confidence_deciles(rows, d, name))
    write_csv(out / "selected_variants_summary.csv", selected_rows)
    write_csv(out / "fusion_per_task.csv", pt_rows)
    write_csv(out / "fusion_confidence_deciles.csv", decile_rows)
    write_csv(out / "adaptive_primary_best_decisions.csv", ba_d)

    # Confusion for best adaptive primary.
    tasks = sorted(set(int(r["gt_task_id"]) for r in rows))
    conf = {t: Counter() for t in tasks}
    for r, d in zip(rows, ba_d):
        conf[int(r["gt_task_id"])][int(d["fusion_winner_task"])] += 1
    conf_rows = []
    for gt in tasks:
        rr = {"gt_task_id": gt}
        rr.update({f"pred_t{p}": conf[gt][p] for p in tasks})
        conf_rows.append(rr)
    write_csv(out / "adaptive_primary_best_confusion.csv", conf_rows)

    summary = {
        "root": str(root),
        "source_logs": [str(p) for p in paths],
        "n": len(rows),
        "score_normalization": "within-Semantic-Top2 pair share: e_k/(e_a+e_b); lower is better, separately for Semantic and Dynamics",
        "semantic_confidence": "normalized_gap=(e2-e1)/(e2+eps), lower means more ambiguous",
        "primary_gate": args.primary_gate,
        "fixed_fusion": "S=(1-lambda)*sem_share + lambda*dyn_share inside the gate; Semantic-only outside",
        "adaptive_fusion": "lambda_dyn=lambda_max*((delta-C)/delta)^gamma clipped to [0,lambda_max] inside gate; 0 outside",
        "semantic_only": sem_s,
        "hard_dynamics_primary_gate": hard_s,
        "best_fixed_at_primary_gate": best_fixed_primary,
        "best_adaptive_at_primary_gate": best_adaptive_primary,
        "diagnostic_global_best": best_global,
        "warning": "All best settings in this output are diagnostic because GT labels from the passive probe are used to select them. Freeze delta/lambda/gamma on a held-out routing validation split before final test evaluation.",
    }
    (out / "fusion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Concise markdown report.
    def pct(x: float) -> str:
        return f"{100.0*float(x):.2f}%"

    report = []
    report.append("# Routing-V2 Confidence-Aware Semantic + Dynamics Fusion\n")
    report.append(f"- Decisions: **{len(rows)}**")
    report.append(f"- Semantic-only: **{pct(sem_s['hybrid_accuracy'])}**")
    report.append(f"- Primary gate: **C_sem < {args.primary_gate:g}**")
    report.append("\n## Best fixed fusion at primary gate\n")
    report.append(f"- lambda_dyn: **{float(best_fixed_primary['lambda_value']):.3f}**")
    report.append(f"- Hybrid: **{pct(best_fixed_primary['hybrid_accuracy'])}** ({100*float(best_fixed_primary['gain_vs_semantic']):+.2f} pp)")
    report.append(f"- Recovery / damage: **{int(best_fixed_primary['recovered'])} / {int(best_fixed_primary['damaged'])}**, net **{int(best_fixed_primary['net_corrections']):+d}**")
    report.append("\n## Best confidence-adaptive fusion at primary gate\n")
    report.append(f"- lambda_max: **{float(best_adaptive_primary['lambda_value']):.3f}**")
    report.append(f"- gamma: **{float(best_adaptive_primary['gamma']):.3f}**")
    report.append(f"- Hybrid: **{pct(best_adaptive_primary['hybrid_accuracy'])}** ({100*float(best_adaptive_primary['gain_vs_semantic']):+.2f} pp)")
    report.append(f"- Recovery / damage: **{int(best_adaptive_primary['recovered'])} / {int(best_adaptive_primary['damaged'])}**, net **{int(best_adaptive_primary['net_corrections']):+d}**")
    report.append(f"- Mean Dynamics weight inside gate: **{float(best_adaptive_primary['mean_lambda_activated']):.3f}**")
    report.append("\n## Diagnostic global best\n")
    report.append(f"- family={best_global['family']}, delta={float(best_global['gate_threshold']):.4f}, lambda={float(best_global['lambda_value']):.3f}, gamma={float(best_global['gamma']):.3f}")
    report.append(f"- Hybrid: **{pct(best_global['hybrid_accuracy'])}** ({100*float(best_global['gain_vs_semantic']):+.2f} pp)")
    report.append("\n> Diagnostic only: do not report these selected hyperparameters as final test settings. Select/freeze gate and fusion weights on a held-out routing validation split.\n")
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("======================================================================")
    print(" Routing-V2 confidence-aware fusion (offline, passive logs only)")
    print(f" Decisions                    : {len(rows)}")
    print(f" Semantic-only                : {100*sem_s['hybrid_accuracy']:.2f}%")
    print(f" Primary gate                 : C_sem < {args.primary_gate:g}")
    print("----------------------------------------------------------------------")
    print(" Best fixed @ primary gate")
    print(f"   lambda_dyn                 : {float(best_fixed_primary['lambda_value']):.3f}")
    print(f"   hybrid                     : {100*float(best_fixed_primary['hybrid_accuracy']):.2f}% ({100*float(best_fixed_primary['gain_vs_semantic']):+.2f} pp)")
    print(f"   recovered/damaged/net      : {int(best_fixed_primary['recovered'])}/{int(best_fixed_primary['damaged'])}/{int(best_fixed_primary['net_corrections']):+d}")
    print(" Best adaptive @ primary gate")
    print(f"   lambda_max / gamma         : {float(best_adaptive_primary['lambda_value']):.3f} / {float(best_adaptive_primary['gamma']):.3f}")
    print(f"   hybrid                     : {100*float(best_adaptive_primary['hybrid_accuracy']):.2f}% ({100*float(best_adaptive_primary['gain_vs_semantic']):+.2f} pp)")
    print(f"   recovered/damaged/net      : {int(best_adaptive_primary['recovered'])}/{int(best_adaptive_primary['damaged'])}/{int(best_adaptive_primary['net_corrections']):+d}")
    print(f"   mean lambda inside gate    : {float(best_adaptive_primary['mean_lambda_activated']):.3f}")
    print("----------------------------------------------------------------------")
    print(f" Output                       : {out}")
    print(" WARNING: best settings are test-set diagnostic only; freeze final hyperparameters on held-out validation.")
    print("======================================================================")


if __name__ == "__main__":
    main()
