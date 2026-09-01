#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Detailed passive diagnosis for Routing-V2 Dynamics verifier. "
            "It never runs the policy; it only re-analyzes existing T*/dynamics_probe.jsonl logs."
        )
    )
    p.add_argument("--root", type=Path, required=True,
                   help="Dynamics probe run root containing T*/dynamics_probe.jsonl")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--confidence-key", default="semantic_normalized_gap")
    p.add_argument("--deciles", type=int, default=10)
    p.add_argument("--threshold-points", type=int, default=201)
    return p.parse_args()


def read_rows(root: Path):
    paths = sorted(root.glob("T*/dynamics_probe.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No T*/dynamics_probe.jsonl under {root}")
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(f"Failed JSON parse {path}:{line_no}: {exc}") from exc
                row["_source_file"] = str(path)
                rows.append(row)
    if not rows:
        raise RuntimeError("No dynamics probe decisions found")
    return rows, paths


def require_fields(rows, fields):
    missing = []
    for field in fields:
        if any(field not in r for r in rows):
            missing.append(field)
    if missing:
        raise KeyError(
            "Missing required dynamics-probe fields: " + ", ".join(missing)
        )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_div(a, b):
    return float(a) / float(b) if b else math.nan


def quantile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    w = pos - lo
    return values[lo] * (1.0 - w) + values[hi] * w


def task_diagnosis(rows):
    out = []
    tasks = sorted(set(int(r["gt_task_id"]) for r in rows))
    for t in tasks:
        rr = [r for r in rows if int(r["gt_task_id"]) == t]
        sem_err = [r for r in rr if not bool(r["semantic_top1_correct"])]
        sem_ok = [r for r in rr if bool(r["semantic_top1_correct"])]
        recovered = sum(bool(r["semantic_error_recovered"]) for r in sem_err)
        damaged = sum(bool(r["semantic_correct_damaged"]) for r in sem_ok)
        sem_correct = sum(bool(r["semantic_top1_correct"]) for r in rr)
        dyn_correct = sum(bool(r["dynamics_correct"]) for r in rr)
        out.append({
            "gt_task_id": t,
            "n": len(rr),
            "semantic_top1_accuracy": safe_div(sem_correct, len(rr)),
            "semantic_top2_recall": safe_div(sum(bool(r["semantic_top2_correct"]) for r in rr), len(rr)),
            "dynamics_top2_accuracy": safe_div(dyn_correct, len(rr)),
            "dynamics_minus_semantic": safe_div(dyn_correct - sem_correct, len(rr)),
            "semantic_errors": len(sem_err),
            "recovered_errors": recovered,
            "recovery_rate": safe_div(recovered, len(sem_err)),
            "semantic_correct": len(sem_ok),
            "damaged_correct": damaged,
            "damage_rate": safe_div(damaged, len(sem_ok)),
            "always_on_net_corrections": recovered - damaged,
            "always_on_net_gain_pp": 100.0 * safe_div(recovered - damaged, len(rr)),
        })
    return out


def confidence_deciles(rows, confidence_key: str, n_bins: int):
    # Equal-count bins after sorting by confidence: bin 0 is most ambiguous.
    ordered = sorted(rows, key=lambda r: float(r[confidence_key]))
    n = len(ordered)
    out = []
    for b in range(n_bins):
        start = (b * n) // n_bins
        end = ((b + 1) * n) // n_bins
        rr = ordered[start:end]
        if not rr:
            continue
        sem_correct = sum(bool(r["semantic_top1_correct"]) for r in rr)
        dyn_correct = sum(bool(r["dynamics_correct"]) for r in rr)
        sem_err = sum(not bool(r["semantic_top1_correct"]) for r in rr)
        recovered = sum(bool(r["semantic_error_recovered"]) for r in rr)
        damaged = sum(bool(r["semantic_correct_damaged"]) for r in rr)
        vals = [float(r[confidence_key]) for r in rr]
        out.append({
            "bin": b,
            "meaning": "lowest confidence" if b == 0 else ("highest confidence" if b == n_bins - 1 else ""),
            "start_rank": start,
            "end_rank_exclusive": end,
            "n": len(rr),
            "confidence_min": min(vals),
            "confidence_mean": sum(vals) / len(vals),
            "confidence_max": max(vals),
            "semantic_accuracy": safe_div(sem_correct, len(rr)),
            "dynamics_accuracy": safe_div(dyn_correct, len(rr)),
            "dynamics_minus_semantic": safe_div(dyn_correct - sem_correct, len(rr)),
            "semantic_errors": sem_err,
            "recovered": recovered,
            "damaged": damaged,
            "net_corrections": recovered - damaged,
            "hybrid_accuracy_if_dyn_on_bin": safe_div(sem_correct + recovered - damaged, len(rr)),
        })
    return out


def prefix_diagnosis(rows, confidence_key: str):
    ordered = sorted(rows, key=lambda r: float(r[confidence_key]))
    n = len(ordered)
    # Dense around the range already indicated by the passive probe, plus broad checkpoints.
    rates = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00,
    ]
    sem_total = sum(bool(r["semantic_top1_correct"]) for r in rows)
    sem_err_total = n - sem_total
    out = []
    for rate in rates:
        k = max(1, min(n, int(round(rate * n))))
        rr = ordered[:k]
        sem_correct_gate = sum(bool(r["semantic_top1_correct"]) for r in rr)
        dyn_correct_gate = sum(bool(r["dynamics_correct"]) for r in rr)
        sem_err_gate = k - sem_correct_gate
        recovered = sum(bool(r["semantic_error_recovered"]) for r in rr)
        damaged = sum(bool(r["semantic_correct_damaged"]) for r in rr)
        hybrid_correct = sem_total + recovered - damaged
        out.append({
            "activation_count": k,
            "activation_rate": k / n,
            "confidence_threshold_inclusive": float(rr[-1][confidence_key]),
            "semantic_accuracy_in_gate": safe_div(sem_correct_gate, k),
            "dynamics_accuracy_in_gate": safe_div(dyn_correct_gate, k),
            "dynamics_minus_semantic_in_gate": safe_div(dyn_correct_gate - sem_correct_gate, k),
            "semantic_errors_gated": sem_err_gate,
            "semantic_error_capture_rate": safe_div(sem_err_gate, sem_err_total),
            "recovered": recovered,
            "damaged": damaged,
            "net_corrections": recovered - damaged,
            "hybrid_correct": hybrid_correct,
            "hybrid_accuracy": hybrid_correct / n,
            "gain_vs_semantic_pp": 100.0 * (hybrid_correct - sem_total) / n,
        })
    return out


def threshold_sweep(rows, confidence_key: str, points: int):
    vals = [float(r[confidence_key]) for r in rows]
    thresholds = sorted(set(
        [min(vals) - 1e-12]
        + [quantile(vals, i / (points - 1)) for i in range(points)]
        + [max(vals) + 1e-12]
    ))
    n = len(rows)
    sem_total = sum(bool(r["semantic_top1_correct"]) for r in rows)
    sem_err_total = n - sem_total
    out = []
    for thr in thresholds:
        gated = [r for r in rows if float(r[confidence_key]) < thr]
        recovered = sum(bool(r["semantic_error_recovered"]) for r in gated)
        damaged = sum(bool(r["semantic_correct_damaged"]) for r in gated)
        sem_err_gate = sum(not bool(r["semantic_top1_correct"]) for r in gated)
        hybrid_correct = sem_total + recovered - damaged
        out.append({
            "threshold": thr,
            "gate_rule": f"{confidence_key} < threshold",
            "activation_count": len(gated),
            "activation_rate": len(gated) / n,
            "semantic_errors_gated": sem_err_gate,
            "semantic_error_capture_rate": safe_div(sem_err_gate, sem_err_total),
            "recovered": recovered,
            "damaged": damaged,
            "net_corrections": recovered - damaged,
            "hybrid_accuracy": hybrid_correct / n,
            "gain_vs_semantic_pp": 100.0 * (hybrid_correct - sem_total) / n,
        })
    return out


def confusion(rows):
    tasks = sorted(set(int(r["gt_task_id"]) for r in rows) | set(int(r["dynamics_winner_task"]) for r in rows))
    table = {t: Counter() for t in tasks}
    for r in rows:
        table[int(r["gt_task_id"])][int(r["dynamics_winner_task"])] += 1
    out = []
    for gt in tasks:
        row = {"gt_task_id": gt}
        for pred in tasks:
            row[f"pred_t{pred}"] = table[gt][pred]
        out.append(row)
    return out


def fmt_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def build_report(rows, task_rows, decile_rows, prefix_rows, sweep_rows, confidence_key, raw_paths):
    n = len(rows)
    sem_correct = sum(bool(r["semantic_top1_correct"]) for r in rows)
    dyn_correct = sum(bool(r["dynamics_correct"]) for r in rows)
    sem_err = [r for r in rows if not bool(r["semantic_top1_correct"])]
    sem_ok = [r for r in rows if bool(r["semantic_top1_correct"])]
    recovered = sum(bool(r["semantic_error_recovered"]) for r in sem_err)
    damaged = sum(bool(r["semantic_correct_damaged"]) for r in sem_ok)
    best = max(sweep_rows, key=lambda r: (r["hybrid_accuracy"], -r["activation_rate"]))
    best_hybrid = float(best["hybrid_accuracy"])

    # How concentrated is the value of dynamics in the lowest-confidence bins?
    positive_bins = [r for r in decile_rows if int(r["net_corrections"]) > 0]
    negative_bins = [r for r in decile_rows if int(r["net_corrections"]) < 0]
    first_nonpositive = next((r for r in decile_rows if float(r["dynamics_minus_semantic"]) <= 0), None)

    lines = []
    lines.append("# Routing-V2 Dynamics diagnosis")
    lines.append("")
    lines.append("This report is passive/offline: no routing decision in this analysis controls the robot.")
    lines.append("")
    lines.append("## Global")
    lines.append("")
    lines.append(f"- Decisions: **{n}**")
    lines.append(f"- Semantic Top-1: **{fmt_pct(sem_correct/n)}**")
    lines.append(f"- Dynamics Top-2 always-on: **{fmt_pct(dyn_correct/n)}**")
    lines.append(f"- Semantic-error recovery: **{recovered}/{len(sem_err)} = {fmt_pct(safe_div(recovered,len(sem_err)))}**")
    lines.append(f"- Semantic-correct damage: **{damaged}/{len(sem_ok)} = {fmt_pct(safe_div(damaged,len(sem_ok)))}**")
    lines.append(f"- Always-on net corrections: **{recovered-damaged:+d}**")
    lines.append("")
    lines.append("## Best diagnostic confidence gate")
    lines.append("")
    lines.append(f"- Metric: `{confidence_key}` (lower = more ambiguous)")
    lines.append(f"- Threshold: **{float(best['threshold']):.8f}**")
    lines.append(f"- Activation: **{100*float(best['activation_rate']):.2f}% ({int(best['activation_count'])}/{n})**")
    lines.append(f"- Net corrections: **{int(best['net_corrections']):+d}**")
    lines.append(f"- Hybrid accuracy: **{100*best_hybrid:.2f}%**")
    lines.append(f"- Gain vs Semantic: **{float(best['gain_vs_semantic_pp']):+.2f} pp**")
    lines.append("")
    lines.append("**Important:** this threshold is test-set diagnostic only. Freeze the final threshold on a held-out routing validation split.")
    lines.append("")
    lines.append("## Per task")
    lines.append("")
    lines.append("| Task | N | Sem Top1 | Dyn Top2 | Recovery | Damage | Always-on net |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in task_rows:
        lines.append(
            f"| T{r['gt_task_id']} | {r['n']} | {fmt_pct(r['semantic_top1_accuracy'])} | "
            f"{fmt_pct(r['dynamics_top2_accuracy'])} | {fmt_pct(r['recovery_rate'])} | "
            f"{fmt_pct(r['damage_rate'])} | {int(r['always_on_net_corrections']):+d} |"
        )
    lines.append("")
    lines.append("## Confidence deciles")
    lines.append("")
    lines.append("Bin 0 is the lowest-confidence / most ambiguous 10% of chunks.")
    lines.append("")
    lines.append("| Bin | N | Semantic | Dynamics | Dyn-Sem | Recovered | Damaged | Net |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in decile_rows:
        lines.append(
            f"| {r['bin']} | {r['n']} | {fmt_pct(r['semantic_accuracy'])} | "
            f"{fmt_pct(r['dynamics_accuracy'])} | {100*float(r['dynamics_minus_semantic']):+.2f} pp | "
            f"{r['recovered']} | {r['damaged']} | {int(r['net_corrections']):+d} |"
        )
    lines.append("")
    if positive_bins:
        lines.append("Positive-net Dynamics bins: " + ", ".join(str(r["bin"]) for r in positive_bins) + ".")
    if negative_bins:
        lines.append("Negative-net Dynamics bins: " + ", ".join(str(r["bin"]) for r in negative_bins) + ".")
    if first_nonpositive is not None:
        lines.append(f"First decile where Dynamics no longer beats Semantic: bin {first_nonpositive['bin']}.")
    lines.append("")
    lines.append("## Interpretation rule")
    lines.append("")
    lines.append("- If Dynamics beats Semantic mainly in the lowest-confidence bins and becomes harmful as confidence rises, confidence-gated WAM verification is supported.")
    lines.append("- If the same few tasks dominate damage, improve or redesign the task-specific Dynamics AE before closed-loop routing.")
    lines.append("- If Dynamics is not better than Semantic even in the lowest-confidence bins, the current `[h, Δh, z]` verifier is not providing useful complementary evidence.")
    lines.append("")
    lines.append("## Source logs")
    for p in raw_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    return "\n".join(lines)


def main():
    a = parse_args()
    root = a.root.expanduser().resolve()
    out = (a.output_dir.expanduser().resolve() if a.output_dir else root / "dynamics_diagnosis")
    out.mkdir(parents=True, exist_ok=True)

    rows, paths = read_rows(root)
    require_fields(rows, [
        "gt_task_id",
        "semantic_top1_task",
        "semantic_top1_correct",
        "semantic_top2_correct",
        "dynamics_winner_task",
        "dynamics_correct",
        "semantic_error_recovered",
        "semantic_correct_damaged",
        a.confidence_key,
    ])

    tasks = task_diagnosis(rows)
    deciles = confidence_deciles(rows, a.confidence_key, max(2, a.deciles))
    prefixes = prefix_diagnosis(rows, a.confidence_key)
    sweep = threshold_sweep(rows, a.confidence_key, max(11, a.threshold_points))
    conf = confusion(rows)

    write_csv(out / "dynamics_task_diagnosis.csv", tasks)
    write_csv(out / "dynamics_confidence_deciles.csv", deciles)
    write_csv(out / "dynamics_low_confidence_prefix.csv", prefixes)
    write_csv(out / "dynamics_threshold_sweep_detailed.csv", sweep)
    write_csv(out / "dynamics_confusion.csv", conf)

    best = max(sweep, key=lambda r: (r["hybrid_accuracy"], -r["activation_rate"]))
    sem_correct = sum(bool(r["semantic_top1_correct"]) for r in rows)
    dyn_correct = sum(bool(r["dynamics_correct"]) for r in rows)
    sem_err = [r for r in rows if not bool(r["semantic_top1_correct"])]
    sem_ok = [r for r in rows if bool(r["semantic_top1_correct"])]
    recovered = sum(bool(r["semantic_error_recovered"]) for r in sem_err)
    damaged = sum(bool(r["semantic_correct_damaged"]) for r in sem_ok)
    summary = {
        "root": str(root),
        "num_decisions": len(rows),
        "confidence_key": a.confidence_key,
        "semantic_top1_accuracy": sem_correct / len(rows),
        "dynamics_top2_accuracy_always_on": dyn_correct / len(rows),
        "semantic_errors": len(sem_err),
        "recovered": recovered,
        "recovery_rate": safe_div(recovered, len(sem_err)),
        "semantic_correct": len(sem_ok),
        "damaged": damaged,
        "damage_rate": safe_div(damaged, len(sem_ok)),
        "always_on_net_corrections": recovered - damaged,
        "best_diagnostic_gate": best,
        "warning": (
            "The best gate uses GT labels from this passive probe and is diagnostic only. "
            "Freeze the final threshold on a held-out routing validation split."
        ),
    }
    (out / "dynamics_diagnosis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = build_report(rows, tasks, deciles, prefixes, sweep, a.confidence_key, paths)
    (out / "REPORT.md").write_text(report, encoding="utf-8")

    print("======================================================================")
    print(" Routing-V2 detailed passive Dynamics diagnosis")
    print(f" root            : {root}")
    print(f" decisions       : {len(rows)}")
    print(f" Semantic Top1   : {100*summary['semantic_top1_accuracy']:.2f}%")
    print(f" Dynamics Top2   : {100*summary['dynamics_top2_accuracy_always_on']:.2f}%")
    print(f" recovery        : {recovered}/{len(sem_err)} = {100*summary['recovery_rate']:.2f}%")
    print(f" damage          : {damaged}/{len(sem_ok)} = {100*summary['damage_rate']:.2f}%")
    print("----------------------------------------------------------------------")
    print(f" best diag gate  : {a.confidence_key} < {float(best['threshold']):.8f}")
    print(f" activation      : {100*float(best['activation_rate']):.2f}%")
    print(f" hybrid accuracy : {100*float(best['hybrid_accuracy']):.2f}%")
    print(f" net corrections : {int(best['net_corrections']):+d}")
    print(f" output          : {out}")
    print("======================================================================")
    print((out / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
