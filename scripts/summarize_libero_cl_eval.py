#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_episodes(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse {path}:{line_no}: {exc}"
                ) from exc

    return records


def summarize_one_run(
    run_dir: Path,
    expected_task_ids: list[int] | None = None,
    expected_trials: int | None = None,
):
    episodes_path = run_dir / "episodes.jsonl"

    if not episodes_path.exists():
        raise FileNotFoundError(
            f"episodes.jsonl not found: {episodes_path}"
        )

    records = load_episodes(episodes_path)

    grouped = defaultdict(list)

    for record in records:
        task_id = int(record["task_id"])
        grouped[task_id].append(record)

    if expected_task_ids is not None:
        actual = sorted(grouped)
        expected = sorted(expected_task_ids)

        if actual != expected:
            raise RuntimeError(
                "Task IDs do not match.\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )

    rows = []

    for task_id in sorted(grouped):
        task_records = grouped[task_id]

        total = len(task_records)
        success = sum(
            int(bool(record["success"]))
            for record in task_records
        )

        success_rate = (
            success / total
            if total > 0
            else 0.0
        )

        description = str(
            task_records[0].get(
                "task_description",
                "",
            )
        )

        task_name = str(
            task_records[0].get(
                "task_name",
                "",
            )
        )

        if (
            expected_trials is not None
            and total != expected_trials
        ):
            raise RuntimeError(
                f"Task {task_id} has {total} episodes, "
                f"expected {expected_trials}."
            )

        rows.append(
            {
                "task_id": task_id,
                "task_description": description,
                "task_name": task_name,
                "successes": success,
                "trials": total,
                "success_rate": success_rate,
            }
        )

    return rows


def write_one_run_summary(
    run_dir: Path,
    rows,
):
    csv_path = run_dir / "per_task_summary.csv"
    json_path = run_dir / "per_task_summary.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "task_description",
                "task_name",
                "successes",
                "trials",
                "success_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return csv_path, json_path


def print_rows(rows):
    print()
    print(
        "Task | Success | Trials | SR      | Description"
    )
    print(
        "-----+---------+--------+---------+"
        "------------------------------------------"
    )

    for row in rows:
        print(
            f"{row['task_id']:>4d} | "
            f"{row['successes']:>7d} | "
            f"{row['trials']:>6d} | "
            f"{row['success_rate']:.4f} | "
            f"{row['task_description']}"
        )

    total_success = sum(
        row["successes"]
        for row in rows
    )

    total_trials = sum(
        row["trials"]
        for row in rows
    )

    overall = (
        total_success / total_trials
        if total_trials
        else 0.0
    )

    mean_task_sr = (
        sum(
            row["success_rate"]
            for row in rows
        ) / len(rows)
        if rows
        else 0.0
    )

    print()
    print(
        f"Total: {total_success}/{total_trials} "
        f"= {overall:.4f}"
    )

    print(
        f"Mean task SR: {mean_task_sr:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
    )

    parser.add_argument(
        "--expected-trials",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    rows = summarize_one_run(
        args.run_dir,
        expected_task_ids=args.task_ids,
        expected_trials=args.expected_trials,
    )

    csv_path, json_path = (
        write_one_run_summary(
            args.run_dir,
            rows,
        )
    )

    print_rows(rows)

    print()
    print("[OK] Saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
