#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read_rows(root: Path):
    paths = sorted(root.glob("T*/semantic_probe.jsonl"))
    if not paths:
        paths = sorted(root.rglob("semantic_probe.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No semantic_probe.jsonl found under {root}")
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(f"Failed to parse {path}:{line_no}: {exc}") from exc
                row["_source"] = str(path)
                rows.append(row)
    if not rows:
        raise RuntimeError("Semantic probe logs contain no decisions.")
    return rows, paths


def pct(x: float) -> float:
    return 100.0 * float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "semantic_summary"
    out.mkdir(parents=True, exist_ok=True)
    rows, paths = read_rows(root)

    bank = sorted({int(t) for r in rows for t in r["bank_tasks"]})
    gt_tasks = sorted({int(r["gt_task_id"]) for r in rows})
    if not set(gt_tasks).issubset(set(bank)):
        raise RuntimeError(f"GT tasks {gt_tasks} are not a subset of bank {bank}")

    groups = defaultdict(list)
    for r in rows:
        groups[int(r["gt_task_id"])].append(r)

    summary_rows = []
    for task in gt_tasks:
        rr = groups[task]
        top1 = sum(bool(x["top1_correct"]) for x in rr)
        top2 = sum(bool(x["top2_correct"]) for x in rr)
        ranks = [int(x["gt_rank"]) for x in rr]
        summary_rows.append({
            "gt_task_id": task,
            "instruction": rr[0].get("instruction", ""),
            "num_chunk_decisions": len(rr),
            "top1_correct": top1,
            "top1_accuracy": top1 / len(rr),
            "top2_correct": top2,
            "top2_recall": top2 / len(rr),
            "mean_gt_rank": mean(ranks),
            "mrr": mean(1.0 / r for r in ranks),
        })

    total = len(rows)
    total_top1 = sum(bool(x["top1_correct"]) for x in rows)
    total_top2 = sum(bool(x["top2_correct"]) for x in rows)
    total_ranks = [int(x["gt_rank"]) for x in rows]
    summary_rows.append({
        "gt_task_id": "ALL",
        "instruction": "",
        "num_chunk_decisions": total,
        "top1_correct": total_top1,
        "top1_accuracy": total_top1 / total,
        "top2_correct": total_top2,
        "top2_recall": total_top2 / total,
        "mean_gt_rank": mean(total_ranks),
        "mrr": mean(1.0 / r for r in total_ranks),
    })

    summary_csv = out / "semantic_topk_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)

    # GT x predicted Top-1 confusion.
    confusion = {g: Counter() for g in gt_tasks}
    for r in rows:
        confusion[int(r["gt_task_id"])][int(r["top1_task"])] += 1
    conf_csv = out / "semantic_top1_confusion.csv"
    with conf_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["gt_task_id"] + [f"pred_t{t}" for t in bank]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for g in gt_tasks:
            row = {"gt_task_id": g}
            row.update({f"pred_t{t}": confusion[g][t] for t in bank})
            w.writerow(row)

    # Mean reconstruction error matrix: GT task x AE task.
    err_csv = out / "semantic_mean_error_matrix.csv"
    with err_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["gt_task_id"] + [f"ae_t{t}" for t in bank]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for g in gt_tasks:
            rr = groups[g]
            row = {"gt_task_id": g}
            for t in bank:
                row[f"ae_t{t}"] = mean(float(x["errors"][str(t)]) for x in rr)
            w.writerow(row)

    pred_counts = Counter(int(r["top1_task"]) for r in rows)
    pred_csv = out / "semantic_top1_selection_frequency.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["expert_task_id", "count", "fraction"])
        w.writeheader()
        for t in bank:
            w.writerow({
                "expert_task_id": t,
                "count": pred_counts[t],
                "fraction": pred_counts[t] / total,
            })

    # Rank histogram per GT task.
    rank_csv = out / "semantic_gt_rank_histogram.csv"
    with rank_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["gt_task_id"] + [f"rank_{r}" for r in range(1, len(bank) + 1)]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for g in gt_tasks:
            c = Counter(int(x["gt_rank"]) for x in groups[g])
            row = {"gt_task_id": g}
            row.update({f"rank_{r}": c[r] for r in range(1, len(bank) + 1)})
            w.writerow(row)

    summary_json = out / "semantic_probe_summary.json"
    payload = {
        "root": str(root),
        "log_files": [str(p) for p in paths],
        "bank_tasks": bank,
        "gt_tasks": gt_tasks,
        "num_chunk_decisions": total,
        "micro_top1_accuracy": total_top1 / total,
        "micro_top2_recall": total_top2 / total,
        "mean_gt_rank": mean(total_ranks),
        "mrr": mean(1.0 / r for r in total_ranks),
        "note": (
            "Passive CL-only semantic probe. Semantic ranking never controls the robot; "
            "each rollout executes the provided task-ID skill checkpoint. Metrics are per action-chunk decision."
        ),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("======================================================================")
    print(" Routing-V2 Semantic AE Bank — passive chunk-level retrieval")
    print(f" Bank: {bank} | decisions={total}")
    print("---------------------------------------------------------------------")
    for row in summary_rows:
        if row["gt_task_id"] == "ALL":
            label = "ALL"
        else:
            label = f"T{row['gt_task_id']}"
        print(
            f" {label:>4s}: n={int(row['num_chunk_decisions']):4d} "
            f"Top1={pct(row['top1_accuracy']):6.2f}% "
            f"Top2={pct(row['top2_recall']):6.2f}% "
            f"MeanRank={float(row['mean_gt_rank']):.3f} "
            f"MRR={float(row['mrr']):.3f}"
        )
    print("---------------------------------------------------------------------")
    print(f" Summary   : {summary_csv}")
    print(f" Confusion : {conf_csv}")
    print(f" Error mat : {err_csv}")
    print(f" Selection : {pred_csv}")
    print("======================================================================")


if __name__ == "__main__":
    main()
