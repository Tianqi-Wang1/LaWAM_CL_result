#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_rows(root: Path):
    for stage in ("CL1", "CL2", "CL3", "CL4"):
        stage_dir = root / stage
        if not stage_dir.is_dir():
            continue
        for path in sorted(stage_dir.glob("T*/routing_chunks.jsonl")):
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        yield stage, path, line_no, json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.expanduser().resolve()

    total = 0
    violations: list[dict] = []
    for stage, path, line_no, row in iter_rows(root):
        total += 1
        checks = {
            "gate_active_false": not bool(row.get("gate_active", False)),
            "lambda_zero": abs(float(row.get("lambda_dyn", 0.0))) <= 1e-12,
            "selected_equals_semantic_top1": int(row["selected_task"]) == int(row["semantic_top1_task"]),
            "routing_correct_matches_semantic": bool(row["routing_correct"]) == bool(row["semantic_top1_correct"]),
            "no_dynamics_scores": not bool(row.get("dynamics_errors_top2")),
            "no_fused_scores": not bool(row.get("fused_scores_top2")),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            violations.append({
                "stage": stage,
                "file": str(path),
                "line": line_no,
                "decision_id": row.get("decision_id"),
                "failed": failed,
            })
            if len(violations) >= 20:
                break

    if total == 0:
        raise RuntimeError(f"No routing rows found under {root}")

    payload = {
        "root": str(root),
        "decisions_checked": total,
        "semantic_only_valid": not violations,
        "violations": violations,
        "invariants": [
            "gate_active == False",
            "lambda_dyn == 0",
            "selected_task == semantic_top1_task",
            "routing_correct == semantic_top1_correct",
            "dynamics_errors_top2 is empty",
            "fused_scores_top2 is empty",
        ],
    }
    out = root / "semantic_only_validation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if violations:
        raise SystemExit(2)
    print(f"[OK] Semantic-only routing validated over {total} decisions -> {out}")


if __name__ == "__main__":
    main()
