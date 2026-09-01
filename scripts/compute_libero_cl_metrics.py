#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import fmean

STAGE_RE = re.compile(r"^CL(\d+)$")
TASK_RE = re.compile(r"^task_(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a LIBERO continual-learning success-rate matrix into "
            "REGEN-style FWT / NBT / AUC metrics. The default protocol supports "
            "a jointly trained Base stage followed by sequential CL tasks."
        )
    )
    p.add_argument(
        "matrices",
        nargs="+",
        type=Path,
        help="One or more SR-matrix CSV files (rows: Base, CL1, ...; columns: task_0, task_1, ...).",
    )
    p.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names, one per input matrix. Defaults to each file stem.",
    )
    p.add_argument(
        "--base-tasks",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Tasks jointly introduced at Base. Default: 0 1 2 3 4 5.",
    )
    p.add_argument(
        "--cl-tasks",
        nargs="+",
        type=int,
        default=[6, 7, 8, 9],
        help="Tasks introduced sequentially at CL1, CL2, ... Default: 6 7 8 9.",
    )
    p.add_argument(
        "--scale",
        choices=["auto", "fraction", "percent"],
        default="auto",
        help=(
            "Input SR scale. 'fraction' expects [0,1], 'percent' expects [0,100], "
            "and 'auto' infers from observed values. Outputs are always percentages."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for outputs. Default: alongside the first input matrix in "
            "a directory named cl_metrics."
        ),
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Threshold used to detect an undefined relative NBT denominator.",
    )
    return p.parse_args()


def read_matrix(path: Path) -> tuple[list[str], list[int], dict[str, dict[int, float]]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{path}: CSV has no header")

        stage_col = next(
            (c for c in ("stage", "Stage", "checkpoint", "Checkpoint") if c in reader.fieldnames),
            None,
        )
        if stage_col is None:
            raise RuntimeError(
                f"{path}: expected a stage column named 'stage' "
                "(or Stage/checkpoint/Checkpoint)"
            )

        task_cols: dict[int, str] = {}
        for col in reader.fieldnames:
            m = TASK_RE.match(col.strip())
            if m:
                task_cols[int(m.group(1))] = col
        if not task_cols:
            raise RuntimeError(f"{path}: no task_N columns found")

        rows = list(reader)

    matrix: dict[str, dict[int, float]] = {}
    for row_idx, row in enumerate(rows, start=2):
        stage = str(row[stage_col]).strip()
        if not stage:
            continue
        if stage in matrix:
            raise RuntimeError(f"{path}:{row_idx}: duplicated stage '{stage}'")

        vals: dict[int, float] = {}
        for task_id, col in task_cols.items():
            raw = str(row.get(col, "")).strip()
            if raw == "":
                vals[task_id] = math.nan
            else:
                try:
                    vals[task_id] = float(raw)
                except ValueError as exc:
                    raise RuntimeError(
                        f"{path}:{row_idx}: cannot parse {col}={raw!r}"
                    ) from exc
        matrix[stage] = vals

    if "Base" not in matrix:
        raise RuntimeError(f"{path}: missing Base row")

    cl_stages = sorted(
        (s for s in matrix if STAGE_RE.match(s)),
        key=lambda s: int(STAGE_RE.match(s).group(1)),
    )
    return ["Base", *cl_stages], sorted(task_cols), matrix


def infer_scale(
    matrix: dict[str, dict[int, float]],
    requested: str,
    path: Path,
) -> float:
    values = [
        value
        for stage_values in matrix.values()
        for value in stage_values.values()
        if not math.isnan(value)
    ]
    if not values:
        raise RuntimeError(f"{path}: matrix contains no numeric SR values")
    if min(values) < -1e-9:
        raise RuntimeError(f"{path}: success rates cannot be negative")

    max_value = max(values)
    if requested == "fraction":
        if max_value > 1.000001:
            raise RuntimeError(f"{path}: --scale fraction but max SR={max_value} > 1")
        return 1.0
    if requested == "percent":
        if max_value > 100.000001:
            raise RuntimeError(f"{path}: --scale percent but max SR={max_value} > 100")
        return 100.0

    if max_value <= 1.000001:
        return 1.0
    if max_value <= 100.000001:
        return 100.0
    raise RuntimeError(f"{path}: cannot infer SR scale because max={max_value} > 100")


def normalize_matrix(
    matrix: dict[str, dict[int, float]],
    divisor: float,
) -> dict[str, dict[int, float]]:
    return {
        stage: {
            task: (value / divisor if not math.isnan(value) else math.nan)
            for task, value in values.items()
        }
        for stage, values in matrix.items()
    }


def validate_protocol(
    path: Path,
    stages: list[str],
    tasks_present: list[int],
    matrix: dict[str, dict[int, float]],
    base_tasks: list[int],
    cl_tasks: list[int],
) -> dict[int, str]:
    if len(set(base_tasks)) != len(base_tasks):
        raise RuntimeError("--base-tasks contains duplicates")
    if len(set(cl_tasks)) != len(cl_tasks):
        raise RuntimeError("--cl-tasks contains duplicates")

    overlap = sorted(set(base_tasks) & set(cl_tasks))
    if overlap:
        raise RuntimeError(f"Base and CL task lists overlap: {overlap}")

    required_tasks = [*base_tasks, *cl_tasks]
    missing_tasks = sorted(set(required_tasks) - set(tasks_present))
    if missing_tasks:
        raise RuntimeError(f"{path}: missing task columns {missing_tasks}")

    expected_cl_stages = [f"CL{i}" for i in range(1, len(cl_tasks) + 1)]
    missing_stages = [s for s in expected_cl_stages if s not in stages]
    if missing_stages:
        raise RuntimeError(f"{path}: missing CL stages {missing_stages}")

    intro_stage = {task: "Base" for task in base_tasks}
    intro_stage.update(
        {task: f"CL{i}" for i, task in enumerate(cl_tasks, start=1)}
    )

    stage_index = {"Base": 0, **{f"CL{i}": i for i in range(1, len(cl_tasks) + 1)}}

    # Seen tasks must have a finite SR from their introduction onward.
    for task, intro in intro_stage.items():
        for idx in range(stage_index[intro], len(cl_tasks) + 1):
            stage = "Base" if idx == 0 else f"CL{idx}"
            value = matrix[stage][task]
            if math.isnan(value):
                raise RuntimeError(
                    f"{path}: task_{task} is missing SR at seen stage {stage}"
                )
            if not (-1e-9 <= value <= 1.000001):
                raise RuntimeError(
                    f"{path}: normalized SR task_{task}/{stage}={value} outside [0,1]"
                )

    return intro_stage


def lifecycle_metrics(
    task: int,
    intro_stage: str,
    matrix: dict[str, dict[int, float]],
    num_cl_stages: int,
    eps: float,
) -> dict[str, object]:
    intro_idx = 0 if intro_stage == "Base" else int(intro_stage[2:])
    stages = [
        "Base" if idx == 0 else f"CL{idx}"
        for idx in range(intro_idx, num_cl_stages + 1)
    ]
    values = [matrix[stage][task] for stage in stages]
    intro_sr = values[0]

    nbt = None
    if len(values) > 1:
        if intro_sr <= eps:
            raise RuntimeError(
                f"task_{task}: introduction SR={intro_sr:.8f}; "
                "REGEN-style relative NBT is undefined at a zero/near-zero denominator"
            )
        nbt = fmean((intro_sr - later) / intro_sr for later in values[1:])

    return {
        "task_id": task,
        "intro_stage": intro_stage,
        "intro_sr": intro_sr,
        "final_sr": values[-1],
        "NBT": nbt,
        "AUC": fmean(values),
        "lifecycle_stages": stages,
        "lifecycle_values": values,
    }


def compute_metrics(
    matrix: dict[str, dict[int, float]],
    base_tasks: list[int],
    cl_tasks: list[int],
    intro_stage: dict[int, str],
    eps: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    all_tasks = [*base_tasks, *cl_tasks]
    num_cl = len(cl_tasks)

    detail = [
        lifecycle_metrics(
            task=task,
            intro_stage=intro_stage[task],
            matrix=matrix,
            num_cl_stages=num_cl,
            eps=eps,
        )
        for task in all_tasks
    ]

    # Primary grouped-Base protocol.
    # Joint Base tasks define initialization and are NOT counted in FWT.
    fwt = fmean(
        matrix[f"CL{i}"][task]
        for i, task in enumerate(cl_tasks, start=1)
    )

    nbt_values = [
        float(row["NBT"])
        for row in detail
        if row["NBT"] is not None
    ]
    nbt = fmean(nbt_values)
    auc = fmean(float(row["AUC"]) for row in detail)

    # Secondary CL-only diagnostic.
    cl_set = set(cl_tasks)
    cl_detail = [row for row in detail if int(row["task_id"]) in cl_set]
    cl_nbt_values = [
        float(row["NBT"])
        for row in cl_detail
        if row["NBT"] is not None
    ]
    cl_only_nbt = fmean(cl_nbt_values) if cl_nbt_values else math.nan
    cl_only_auc = fmean(float(row["AUC"]) for row in cl_detail)

    final_stage = f"CL{num_cl}"
    final_seen = fmean(matrix[final_stage][task] for task in all_tasks)
    final_base = fmean(matrix[final_stage][task] for task in base_tasks)
    final_cl = fmean(matrix[final_stage][task] for task in cl_tasks)

    return (
        {
            "FWT": 100.0 * fwt,
            "NBT": 100.0 * nbt,
            "AUC": 100.0 * auc,
            "CL_only_FWT": 100.0 * fwt,
            "CL_only_NBT": 100.0 * cl_only_nbt,
            "CL_only_AUC": 100.0 * cl_only_auc,
            "final_seen_SR": 100.0 * final_seen,
            "final_base_SR": 100.0 * final_base,
            "final_cl_SR": 100.0 * final_cl,
        },
        detail,
    )


def write_task_detail(
    path: Path,
    method: str,
    detail: list[dict[str, object]],
) -> None:
    fields = [
        "method",
        "task_id",
        "intro_stage",
        "intro_sr_percent",
        "final_sr_percent",
        "NBT_percent",
        "AUC_percent",
        "lifecycle_stages",
        "lifecycle_sr_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in detail:
            nbt = row["NBT"]
            writer.writerow(
                {
                    "method": method,
                    "task_id": row["task_id"],
                    "intro_stage": row["intro_stage"],
                    "intro_sr_percent": f"{100.0 * float(row['intro_sr']):.6f}",
                    "final_sr_percent": f"{100.0 * float(row['final_sr']):.6f}",
                    "NBT_percent": (
                        "" if nbt is None else f"{100.0 * float(nbt):.6f}"
                    ),
                    "AUC_percent": f"{100.0 * float(row['AUC']):.6f}",
                    "lifecycle_stages": "|".join(row["lifecycle_stages"]),
                    "lifecycle_sr_percent": "|".join(
                        f"{100.0 * float(v):.6f}"
                        for v in row["lifecycle_values"]
                    ),
                }
            )


def main() -> int:
    args = parse_args()

    if args.names is not None and len(args.names) not in (0, len(args.matrices)):
        raise RuntimeError(
            f"--names must contain exactly {len(args.matrices)} entries"
        )
    names = (
        [path.stem for path in args.matrices]
        if not args.names
        else list(args.names)
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.matrices[0].expanduser().resolve().parent / "cl_metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    run_meta = []

    for matrix_path, method in zip(args.matrices, names):
        matrix_path = matrix_path.expanduser().resolve()
        stages, tasks_present, raw_matrix = read_matrix(matrix_path)
        divisor = infer_scale(raw_matrix, args.scale, matrix_path)
        matrix = normalize_matrix(raw_matrix, divisor)

        intro_stage = validate_protocol(
            matrix_path,
            stages,
            tasks_present,
            matrix,
            args.base_tasks,
            args.cl_tasks,
        )
        metrics, detail = compute_metrics(
            matrix,
            args.base_tasks,
            args.cl_tasks,
            intro_stage,
            args.eps,
        )

        summary_rows.append(
            {
                "method": method,
                "input_file": str(matrix_path),
                **metrics,
            }
        )

        detail_path = output_dir / f"{matrix_path.stem}_task_metrics.csv"
        write_task_detail(detail_path, method, detail)

        run_meta.append(
            {
                "method": method,
                "input_file": str(matrix_path),
                "detected_input_scale": "fraction" if divisor == 1.0 else "percent",
                "metrics_percent": metrics,
                "task_detail_file": str(detail_path),
            }
        )

        print(
            f"{method:24s} "
            f"FWT={metrics['FWT']:.2f}  "
            f"NBT={metrics['NBT']:.2f}  "
            f"AUC={metrics['AUC']:.2f}  "
            f"| CL-only NBT={metrics['CL_only_NBT']:.2f}  "
            f"AUC={metrics['CL_only_AUC']:.2f}"
        )

    summary_path = output_dir / "cl_metrics_summary.csv"
    fields = [
        "method",
        "input_file",
        "FWT",
        "NBT",
        "AUC",
        "CL_only_FWT",
        "CL_only_NBT",
        "CL_only_AUC",
        "final_seen_SR",
        "final_base_SR",
        "final_cl_SR",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "protocol": {
            "name": "REGEN-style grouped-Base",
            "base_tasks": args.base_tasks,
            "cl_tasks": args.cl_tasks,
            "FWT": (
                "Mean SR of sequential CL tasks at their introduction stage. "
                "Joint Base tasks are initialization and are not counted in FWT."
            ),
            "NBT": (
                "Per task: mean relative degradation (r_intro-r_later)/r_intro over "
                "all later stages. Then macro-average over every task with a later stage. "
                "Joint Base tasks use Base as r_intro."
            ),
            "AUC": (
                "Per task: mean SR from its introduction stage through the final stage. "
                "Then macro-average over all Base and CL tasks."
            ),
            "CL_only": (
                "Secondary diagnostic using only sequential CL tasks; FWT is unchanged."
            ),
            "output_scale": "percent",
        },
        "runs": run_meta,
    }
    meta_path = output_dir / "cl_metrics_meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSummary : {summary_path}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise