#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def read_rows(root: Path) -> list[dict]:
    paths = sorted(root.glob("T*/semantic_probe.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No T*/semantic_probe.jsonl under {root}")
    rows: list[dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_source_file"] = str(p)
                    rows.append(row)
    if not rows:
        raise RuntimeError(f"No semantic decisions under {root}")
    return rows


def confidence_metrics(row: dict, eps: float = 1e-12) -> dict[str, float]:
    ranked = [int(x) for x in row["ranked_tasks"]]
    errors = {int(k): float(v) for k, v in row["errors"].items()}
    e1 = errors[ranked[0]]
    e2 = errors[ranked[1]]
    gap = e2 - e1
    rel = gap / max(abs(e1), eps)
    # Bounded [0,1] for positive reconstruction errors; higher = more confident.
    norm_gap = gap / max(abs(e2), eps)
    ratio = e2 / max(abs(e1), eps)
    return {
        "top1_error": e1,
        "top2_error": e2,
        "abs_margin": gap,
        "relative_margin": rel,
        "normalized_gap": norm_gap,
        "error_ratio": ratio,
    }


def quantile(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    x = (len(s) - 1) * q
    lo = int(math.floor(x)); hi = int(math.ceil(x))
    if lo == hi:
        return s[lo]
    w = x - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def make_sweep(rows: list[dict], metric: str, points: int) -> list[dict]:
    vals = [float(r[metric]) for r in rows]
    # Gate Dynamics when semantic confidence < threshold.
    qs = [i / max(1, points - 1) for i in range(points)]
    thresholds = sorted(set([min(vals) - 1e-12] + [quantile(vals, q) for q in qs] + [max(vals) + 1e-12]))
    n = len(rows)
    n_err = sum(not bool(r["top1_correct"]) for r in rows)
    out = []
    for thr in thresholds:
        gated = [r for r in rows if float(r[metric]) < thr]
        ungated = [r for r in rows if float(r[metric]) >= thr]
        gated_err = sum(not bool(r["top1_correct"]) for r in gated)
        missed_err = n_err - gated_err
        ungated_correct = sum(bool(r["top1_correct"]) for r in ungated)
        out.append({
            "metric": metric,
            "threshold": thr,
            "gate_rule": f"{metric} < threshold",
            "activation_count": len(gated),
            "activation_rate": len(gated) / n,
            "semantic_errors_total": n_err,
            "semantic_errors_captured": gated_err,
            "semantic_error_capture_rate": (gated_err / n_err) if n_err else 1.0,
            "gated_error_precision": (gated_err / len(gated)) if gated else 0.0,
            "ungated_count": len(ungated),
            "ungated_semantic_accuracy": (ungated_correct / len(ungated)) if ungated else 1.0,
            "missed_semantic_errors": missed_err,
            "oracle_upper_bound_if_gated_dynamics_perfect": 1.0 - missed_err / n,
        })
    return out


def choose_min_activation(sweep: list[dict], *, capture: float | None = None, ungated_acc: float | None = None):
    cand = []
    for r in sweep:
        if capture is not None and float(r["semantic_error_capture_rate"]) + 1e-12 < capture:
            continue
        if ungated_acc is not None and float(r["ungated_semantic_accuracy"]) + 1e-12 < ungated_acc:
            continue
        cand.append(r)
    if not cand:
        return None
    return min(cand, key=lambda r: (float(r["activation_rate"]), float(r["threshold"])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--sweep-points", type=int, default=101)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "semantic_confidence"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root)

    decision_rows = []
    for i, r in enumerate(rows):
        c = confidence_metrics(r)
        decision_rows.append({
            "global_index": i,
            "gt_task_id": int(r["gt_task_id"]),
            "top1_task": int(r["top1_task"]),
            "top1_correct": bool(r["top1_correct"]),
            "gt_rank": int(r["gt_rank"]),
            **c,
        })
    write_csv(out / "semantic_confidence_decisions.csv", decision_rows)

    metrics = ["normalized_gap", "relative_margin", "abs_margin", "error_ratio"]
    summary = []
    for metric in metrics:
        allv = [float(r[metric]) for r in decision_rows]
        okv = [float(r[metric]) for r in decision_rows if bool(r["top1_correct"])]
        badv = [float(r[metric]) for r in decision_rows if not bool(r["top1_correct"])]
        summary.append({
            "metric": metric,
            "n": len(allv),
            "mean_all": mean(allv),
            "mean_correct": mean(okv) if okv else math.nan,
            "mean_wrong": mean(badv) if badv else math.nan,
            "median_correct": quantile(okv, 0.5) if okv else math.nan,
            "median_wrong": quantile(badv, 0.5) if badv else math.nan,
            "q10_correct": quantile(okv, 0.1) if okv else math.nan,
            "q90_wrong": quantile(badv, 0.9) if badv else math.nan,
        })
    write_csv(out / "semantic_confidence_summary.csv", summary)

    all_sweeps = []
    recommendations = {}
    for metric in metrics:
        sweep = make_sweep(decision_rows, metric, max(11, int(args.sweep_points)))
        all_sweeps.extend(sweep)
        recommendations[metric] = {
            "min_activation_for_90pct_error_capture": choose_min_activation(sweep, capture=0.90),
            "min_activation_for_95pct_error_capture": choose_min_activation(sweep, capture=0.95),
            "min_activation_for_99pct_ungated_accuracy": choose_min_activation(sweep, ungated_acc=0.99),
        }
    write_csv(out / "semantic_confidence_threshold_sweep.csv", all_sweeps)

    # Primary metric bins: normalized gap, low confidence -> high confidence.
    metric = "normalized_gap"
    vals = [float(r[metric]) for r in decision_rows]
    edges = [quantile(vals, i / 10.0) for i in range(11)]
    bins = []
    for b in range(10):
        lo, hi = edges[b], edges[b + 1]
        if b == 9:
            rr = [r for r in decision_rows if float(r[metric]) >= lo and float(r[metric]) <= hi]
        else:
            rr = [r for r in decision_rows if float(r[metric]) >= lo and float(r[metric]) < hi]
        if not rr:
            continue
        bins.append({
            "bin": b,
            "metric": metric,
            "lo": lo,
            "hi": hi,
            "n": len(rr),
            "semantic_top1_accuracy": sum(bool(r["top1_correct"]) for r in rr) / len(rr),
            "semantic_error_count": sum(not bool(r["top1_correct"]) for r in rr),
        })
    write_csv(out / "semantic_confidence_decile_bins.csv", bins)

    payload = {
        "root": str(root),
        "output_dir": str(out),
        "num_decisions": len(decision_rows),
        "semantic_top1_accuracy": sum(bool(r["top1_correct"]) for r in decision_rows) / len(decision_rows),
        "primary_confidence_metric": "normalized_gap = (e2-e1)/(e2+eps), higher means more confident",
        "gate_rule": "invoke Dynamics AE when normalized_gap < threshold",
        "recommendations_are_diagnostic_only": True,
        "warning": (
            "Thresholds in this file use GT correctness from the passive evaluation and must not be tuned on the final test set. "
            "Use them to understand separability, then choose/freeze the operating threshold on a held-out routing validation split."
        ),
        "recommendations": recommendations,
    }
    (out / "semantic_confidence_recommendations.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("======================================================================")
    print(" Routing-V2 Semantic confidence analysis")
    print(f" Root      : {root}")
    print(f" Decisions : {len(decision_rows)}")
    print(f" Top1      : {100.0 * payload['semantic_top1_accuracy']:.2f}%")
    print(" Primary   : normalized_gap=(e2-e1)/e2; low -> ambiguous -> Dynamics")
    print("---------------------------------------------------------------------")
    rec = recommendations["normalized_gap"]
    for name, row in rec.items():
        if row is None:
            print(f" {name}: unavailable")
        else:
            print(
                f" {name}: threshold={float(row['threshold']):.6g} "
                f"activate={100.0*float(row['activation_rate']):.2f}% "
                f"capture={100.0*float(row['semantic_error_capture_rate']):.2f}% "
                f"ungated_acc={100.0*float(row['ungated_semantic_accuracy']):.2f}%"
            )
    print(f" Output    : {out}")
    print("======================================================================")


if __name__ == "__main__":
    main()
