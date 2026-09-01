#!/usr/bin/env python3
"""Compare pretrained-init and scratch-init Full-Flow T9 summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_task(path: Path, task_id: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["task_id"]) == task_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one task_id={task_id} row in {path}, "
            f"found {len(matches)}"
        )
    row = matches[0]
    return {
        "task_id": task_id,
        "description": row.get("task_description", ""),
        "successes": int(row["successes"]),
        "trials": int(row["trials"]),
        "success_rate": float(row["success_rate"]),
        "path": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=9)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    pretrained = load_task(args.pretrained, args.task_id)
    scratch = load_task(args.scratch, args.task_id)
    if pretrained["trials"] != scratch["trials"]:
        raise RuntimeError(
            "The two controls use different trial counts: "
            f"pretrained={pretrained['trials']}, scratch={scratch['trials']}"
        )

    pretrained_sr = float(pretrained["success_rate"])
    scratch_sr = float(scratch["success_rate"])
    delta = pretrained_sr - scratch_sr
    retained = scratch_sr / pretrained_sr if pretrained_sr > 0 else None
    report = {
        "task_id": args.task_id,
        "pretrained": pretrained,
        "scratch": scratch,
        "pretrained_minus_scratch_sr": delta,
        "scratch_retained_fraction": retained,
    }

    print("Initialization | Success | Trials | SR")
    print("---------------+---------+--------+--------")
    for label, row in (("Pretrained", pretrained), ("Scratch", scratch)):
        print(
            f"{label:>14} | {int(row['successes']):>7} | "
            f"{int(row['trials']):>6} | {float(row['success_rate']):.4f}"
        )
    print()
    print(f"Pretrained - Scratch SR: {delta:+.4f}")
    if retained is not None:
        print(f"Scratch retained fraction: {retained:.4f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"[OK] Saved: {args.output_json}")


if __name__ == "__main__":
    main()
