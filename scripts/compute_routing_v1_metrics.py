#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--protocol", choices=["cl_only", "base_inclusive"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def load(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        rows = list(r)
    tasks = sorted(int(x[5:]) for x in fields if x.startswith("task_"))
    M: dict[str, dict[int, float]] = {}
    for row in rows:
        st = row["stage"].strip()
        if st in M:
            raise RuntimeError(f"Duplicate stage {st} in {path}")
        M[st] = {}
        for t in tasks:
            raw = row.get(f"task_{t}", "").strip()
            M[st][t] = math.nan if raw == "" else float(raw)
    return tasks, M


def nbt(vals: list[float], eps: float):
    intro = vals[0]
    if len(vals) <= 1:
        return None
    if intro <= eps:
        return math.nan
    return fmean((intro - x) / intro for x in vals[1:])


def finish_metrics(metrics: dict[str, float], detail: list[dict[str, Any]]) -> dict[str, Any]:
    nbt_defined = [d for d in detail if d["NBT"] is not None and not math.isnan(d["NBT"])]
    nbt_undefined = [d["task_id"] for d in detail if d["NBT"] is not None and math.isnan(d["NBT"])]
    metrics = dict(metrics)
    metrics["NBT_valid_tasks"] = len(nbt_defined)
    metrics["NBT_undefined_task_ids"] = nbt_undefined
    return metrics


def cl_only(M, eps):
    tasks = [6, 7, 8, 9]
    stages = ["CL1", "CL2", "CL3", "CL4"]
    for i, (t, st) in enumerate(zip(tasks, stages)):
        if st not in M:
            raise RuntimeError(f"missing stage {st}")
        for later in stages[i:]:
            if later not in M or math.isnan(M[later][t]):
                raise RuntimeError(f"missing task_{t} at {later}")
    diag = [M[st][t] for st, t in zip(stages, tasks)]
    detail = []
    for i, t in enumerate(tasks):
        vals = [M[st][t] for st in stages[i:]]
        detail.append({"task_id": t, "intro_stage": stages[i], "values": vals, "NBT": nbt(vals, eps), "AUC": fmean(vals)})
    n = [d["NBT"] for d in detail if d["NBT"] is not None and not math.isnan(d["NBT"])]
    metrics = {
        "FWT": 100 * fmean(diag),
        "NBT": 100 * fmean(n) if n else math.nan,
        "AUC": 100 * fmean(d["AUC"] for d in detail),
        "final_seen_SR": 100 * fmean(M["CL4"][t] for t in tasks),
    }
    return finish_metrics(metrics, detail), detail


def base_inclusive(M, eps):
    base = list(range(6))
    cl = [6, 7, 8, 9]
    stages = ["Base", "CL1", "CL2", "CL3", "CL4"]
    for st in stages:
        if st not in M:
            raise RuntimeError(f"missing stage {st}")
    intro = {**{t: 0 for t in base}, **{t: i + 1 for i, t in enumerate(cl)}}
    detail = []
    for t in base + cl:
        i = intro[t]
        vals = [M[st][t] for st in stages[i:]]
        if any(math.isnan(x) for x in vals):
            raise RuntimeError(f"missing lifecycle SR for task_{t}")
        detail.append({"task_id": t, "intro_stage": stages[i], "values": vals, "NBT": nbt(vals, eps), "AUC": fmean(vals)})

    # Grouped-acquisition extension: T0..T5 are acquired jointly at Base, while
    # T6..T9 are introduced one at a time. This is intentionally not a 10-stage sequence.
    fwt_all = fmean(M[stages[intro[t]]][t] for t in base + cl)
    fwt_cl = fmean(M[f"CL{i}"][t] for i, t in enumerate(cl, start=1))
    n = [d["NBT"] for d in detail if d["NBT"] is not None and not math.isnan(d["NBT"])]
    metrics = {
        "FWT": 100 * fwt_all,
        "FWT_all_acquisition": 100 * fwt_all,
        "FWT_CL_tasks": 100 * fwt_cl,
        "NBT": 100 * fmean(n) if n else math.nan,
        "AUC": 100 * fmean(d["AUC"] for d in detail),
        "final_seen_SR": 100 * fmean(M["CL4"][t] for t in base + cl),
        "final_base_SR": 100 * fmean(M["CL4"][t] for t in base),
        "final_cl_SR": 100 * fmean(M["CL4"][t] for t in cl),
    }
    return finish_metrics(metrics, detail), detail


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main() -> None:
    a = args()
    tasks, M = load(a.matrix)
    if a.protocol == "cl_only":
        if tasks != [6, 7, 8, 9]:
            raise RuntimeError(f"cl_only expects task_6..task_9, got {tasks}")
        metrics, detail = cl_only(M, a.eps)
        note = "Standard four-stage CL-only lifecycle; Base expert/tasks are absent from both routing competition and metrics."
    else:
        if tasks != list(range(10)):
            raise RuntimeError(f"base_inclusive expects task_0..task_9, got {tasks}")
        metrics, detail = base_inclusive(M, a.eps)
        note = "Grouped-acquisition lifecycle: T0..T5 are jointly acquired at Base; T6..T9 are introduced at CL1..CL4."

    a.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": a.protocol,
        "note": note,
        "metrics_percent": json_safe(metrics),
    }
    (a.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (a.output_dir / "task_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "intro_stage", "intro_sr", "final_sr", "NBT", "AUC", "lifecycle"])
        for d in detail:
            nbt_value = d["NBT"]
            w.writerow([
                d["task_id"],
                d["intro_stage"],
                100 * d["values"][0],
                100 * d["values"][-1],
                "" if nbt_value is None or (isinstance(nbt_value, float) and math.isnan(nbt_value)) else 100 * nbt_value,
                100 * d["AUC"],
                "|".join(f"{100*x:.4f}" for x in d["values"]),
            ])

    print(f"[RoutingV1 metrics] protocol={a.protocol}")
    print(f"  note: {note}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if not math.isnan(v) else f"  {k}: n/a")
        else:
            print(f"  {k}: {v}")
    print(f"  output: {a.output_dir}")


if __name__ == "__main__":
    main()
