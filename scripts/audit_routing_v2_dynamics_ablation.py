#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-root", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", type=int, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    missing = []
    for variant in args.variants:
        for task in args.tasks:
            root = args.v2_root / f"task{task}" / "routing_memory_variants" / variant
            pt = root / "dynamics_ae.pt"
            js = root / "dynamics_ae.json"
            if not pt.is_file() or not js.is_file():
                missing.append(str(pt))
                continue
            meta = json.loads(js.read_text(encoding="utf-8"))
            try:
                sd = torch.load(pt, map_location="cpu", weights_only=True)
            except TypeError:
                sd = torch.load(pt, map_location="cpu")
            actual = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
            rows.append({
                "variant": variant,
                "task": task,
                "wm_source": meta.get("wm_source"),
                "input_mode": meta.get("input_mode"),
                "num_tensors": len(sd),
                "num_params": actual,
                "path": str(pt.resolve()),
            })
    if missing:
        raise SystemExit("Missing ablation outputs:\n" + "\n".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "task", "wm_source", "input_mode", "num_tensors", "num_params", "path"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"[OK] audited {len(rows)} Dynamics memories -> {args.output}")
    for variant in args.variants:
        sub = [r for r in rows if r["variant"] == variant]
        params = sorted({int(r["num_params"]) for r in sub})
        print(f"  {variant:14s} tasks={len(sub)} params={params}")


if __name__ == "__main__":
    main()
