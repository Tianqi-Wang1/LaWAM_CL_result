#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED = {
    "cl_only": {
        "stages": ["CL1", "CL2", "CL3", "CL4"],
        "tasks": [6, 7, 8, 9],
        "stage_tasks": {
            "CL1": [6],
            "CL2": [6, 7],
            "CL3": [6, 7, 8],
            "CL4": [6, 7, 8, 9],
        },
    },
    "base_inclusive": {
        "stages": ["Base", "CL1", "CL2", "CL3", "CL4"],
        "tasks": list(range(10)),
        "stage_tasks": {
            "Base": [0, 1, 2, 3, 4, 5],
            "CL1": [0, 1, 2, 3, 4, 5, 6],
            "CL2": [0, 1, 2, 3, 4, 5, 6, 7],
            "CL3": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "CL4": list(range(10)),
        },
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True, help="CSV columns: stage,candidates,task_ids,summary_csv,...")
    p.add_argument("--tasks", nargs="+", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", choices=["cl_only", "base_inclusive"], default=None)
    return p.parse_args()


def parse_task_ids(raw: str) -> list[int]:
    raw = (raw or "").strip()
    return [] if not raw else [int(x) for x in raw.replace(",", " ").split()]


def read_summary(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"task_id", "success_rate"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"Summary {path} missing columns: {sorted(missing)}")
        for row in r:
            task = int(row["task_id"])
            if task in out:
                raise RuntimeError(f"Duplicate task_id={task} in {path}")
            out[task] = float(row["success_rate"])
    if not out:
        raise RuntimeError(f"Empty per-task summary: {path}")
    return out


def infer_protocol(tasks: list[int]) -> str | None:
    if tasks == [6, 7, 8, 9]:
        return "cl_only"
    if tasks == list(range(10)):
        return "base_inclusive"
    return None


def main() -> None:
    a = parse_args()
    tasks = list(a.tasks)
    protocol = a.protocol or infer_protocol(tasks)
    if protocol is not None and tasks != EXPECTED[protocol]["tasks"]:
        raise RuntimeError(
            f"{protocol} expects tasks={EXPECTED[protocol]['tasks']}, got {tasks}"
        )

    rows: list[tuple[str, dict[int, float]]] = []
    seen_stages: set[str] = set()
    with a.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        required = {"stage", "summary_csv"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"Manifest {a.manifest} missing columns: {sorted(missing)}")

        for line_no, row in enumerate(r, start=2):
            # csv.DictReader stores surplus columns under key None. This catches the
            # historical bug where candidates='t6,t7' was written without CSV quoting.
            if None in row and row[None]:
                raise RuntimeError(
                    f"Malformed manifest row {line_no}: extra CSV columns={row[None]!r}. "
                    "A field containing commas was likely written without CSV quoting. "
                    "Rebuild the manifest with csv.writer or use postprocess_routing_v1_existing.sh."
                )
            stage = (row.get("stage") or "").strip()
            if not stage:
                raise RuntimeError(f"Manifest row {line_no} has empty stage")
            if stage in seen_stages:
                raise RuntimeError(f"Duplicate stage {stage!r} in manifest")
            seen_stages.add(stage)

            raw_summary = (row.get("summary_csv") or "").strip()
            if not raw_summary:
                raise RuntimeError(f"Manifest row {line_no} stage={stage} has empty summary_csv")
            summary_path = Path(raw_summary).expanduser().resolve()
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Manifest stage={stage} summary_csv does not exist: {summary_path}"
                )
            vals = read_summary(summary_path)

            if protocol is not None:
                expected_stage_tasks = EXPECTED[protocol]["stage_tasks"].get(stage)
                if expected_stage_tasks is None:
                    raise RuntimeError(
                        f"Unexpected stage {stage!r} for protocol={protocol}; "
                        f"expected {EXPECTED[protocol]['stages']}"
                    )
                manifest_tasks = parse_task_ids(row.get("task_ids") or "")
                if manifest_tasks and manifest_tasks != expected_stage_tasks:
                    raise RuntimeError(
                        f"Stage {stage} manifest task_ids={manifest_tasks}, "
                        f"expected {expected_stage_tasks}"
                    )
                missing_tasks = [t for t in expected_stage_tasks if t not in vals]
                if missing_tasks:
                    raise RuntimeError(
                        f"Stage {stage} summary {summary_path} missing expected tasks {missing_tasks}; "
                        f"contains {sorted(vals)}"
                    )
            rows.append((stage, vals))

    if not rows:
        raise RuntimeError("Empty manifest")

    if protocol is not None:
        actual = [stage for stage, _ in rows]
        expected = EXPECTED[protocol]["stages"]
        if actual != expected:
            raise RuntimeError(
                f"Manifest stage order/content mismatch for {protocol}: got {actual}, expected {expected}"
            )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage", *[f"task_{t}" for t in tasks]]
    with a.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for stage, vals in rows:
            w.writerow(
                {
                    "stage": stage,
                    **{
                        f"task_{t}": "" if t not in vals else f"{vals[t]:.8f}"
                        for t in tasks
                    },
                }
            )

    print(f"[RoutingV1] manifest validation: PASS ({len(rows)} stages, protocol={protocol or 'custom'})")
    print(f"[RoutingV1] SR matrix: {a.output}")
    with a.output.open("r", encoding="utf-8") as f:
        print(f.read().rstrip())


if __name__ == "__main__":
    main()
