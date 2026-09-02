#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(paths):
    rows = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarize_subset(scope, subset):
    n = len(subset)
    sem = sum(bool(r["semantic_top1_correct"]) for r in subset)
    route = sum(bool(r["routing_correct"]) for r in subset)
    gated = sum(bool(r["gate_active"]) for r in subset)
    sem_errors = n - sem
    gated_sem_errors = sum(bool(r["gate_active"]) and (not bool(r["semantic_top1_correct"])) for r in subset)
    recovered = sum((not bool(r["semantic_top1_correct"])) and bool(r["routing_correct"]) for r in subset)
    damaged = sum(bool(r["semantic_top1_correct"]) and (not bool(r["routing_correct"])) for r in subset)
    act_lams = [float(r["lambda_dyn"]) for r in subset if bool(r["gate_active"])]
    return {
        "scope": scope,
        "decisions": n,
        "semantic_top1_accuracy": sem / n,
        "routing_accuracy": route / n,
        "routing_gain_vs_semantic_pp": 100.0 * (route - sem) / n,
        "gate_activation_rate": gated / n,
        "semantic_errors": sem_errors,
        "semantic_errors_gated": gated_sem_errors,
        "semantic_error_capture_rate": (gated_sem_errors / sem_errors) if sem_errors else None,
        "fusion_recovered": recovered,
        "fusion_recovery_rate": (recovered / sem_errors) if sem_errors else None,
        "fusion_damaged": damaged,
        "fusion_damage_rate": (damaged / sem) if sem else None,
        "fusion_net_corrections": recovered - damaged,
        "mean_lambda_dyn": sum(float(r["lambda_dyn"]) for r in subset) / n,
        "mean_lambda_activated": (sum(act_lams) / len(act_lams)) if act_lams else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    paths = [a.stage_dir / f"T{t}" / "routing_chunks.jsonl" for t in a.tasks]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    rows = read_rows(paths)
    if not rows:
        raise RuntimeError("no routing rows")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[int(r["gt_task_id"])].append(r)

    summary = [summarize_subset("ALL", rows)]
    for t in a.tasks:
        if groups[t]:
            summary.append(summarize_subset(f"T{t}", groups[t]))

    with (a.output_dir / "routing_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    counts = Counter((int(r["gt_task_id"]), int(r["selected_task"])) for r in rows)
    candidates = sorted({int(t) for r in rows for t in r["candidate_tasks"]})
    with (a.output_dir / "routing_confusion.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["gt_task"] + [f"pred_T{t}" for t in candidates]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for gt in a.tasks:
            w.writerow({"gt_task": gt, **{f"pred_T{t}": counts[(gt, t)] for t in candidates}})

    payload = {
        "variant": "basewm_hdh_harddyn",
        "decision_rule": "semantic_top1_else_hard_dynamics",
        "routing_wm_source": "base",
        "dynamics_input_mode": "hdh",
        "decisions": len(rows),
        "summary": summary,
        "candidate_tasks": candidates,
    }
    (a.output_dir / "routing_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[RoutingV2][B2-HARDDYN] routing summary -> {a.output_dir}")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
