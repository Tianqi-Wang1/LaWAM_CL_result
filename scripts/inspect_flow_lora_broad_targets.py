#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

FLOW_PREFIX = "policy_backend.flow."

TRANSFORMER_PATTERNS = [
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_(q|k|v)\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_out\.0\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.ff\.net\.0\.proj\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.ff\.net\.2\.weight$"),
]


def load_state(path: Path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint: {path}")
    return obj


def lora_numel(weight: torch.Tensor, rank: int) -> int:
    if weight.ndim != 2:
        raise ValueError("LoRA estimator expects 2-D Linear weight")
    out_features, in_features = map(int, weight.shape)
    return rank * (in_features + out_features)


def classify(suffix: str):
    if any(p.match(suffix) for p in TRANSFORMER_PATTERNS):
        return "transformer"
    if suffix.startswith("enc_vlm.") and suffix.endswith(".weight"):
        return "enc_vlm"
    if suffix.startswith("enc_wm.") and suffix.endswith(".weight"):
        return "enc_wm"
    if suffix in ("DiT.proj_out_1.weight", "DiT.proj_out_2.weight"):
        return "output"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--rank", type=int, default=8)
    args = p.parse_args()
    if args.rank <= 0:
        raise ValueError("rank must be > 0")

    sd = load_state(args.checkpoint)
    groups = {g: [] for g in ("transformer", "enc_vlm", "enc_wm", "output")}

    for key, tensor in sd.items():
        if not torch.is_tensor(tensor) or not str(key).startswith(FLOW_PREFIX):
            continue
        suffix = str(key)[len(FLOW_PREFIX):]
        group = classify(suffix)
        if group is not None and tensor.ndim == 2:
            groups[group].append((suffix, tensor))

    print("=" * 100)
    print(f"LaWAM Broad Flow-LoRA target inspection | rank={args.rank}")
    print("=" * 100)
    total = 0
    for group in ("transformer", "enc_vlm", "enc_wm", "output"):
        items = sorted(groups[group], key=lambda x: x[0])
        params = sum(lora_numel(t, args.rank) for _, t in items)
        total += params
        print(f"\n[{group}] modules={len(items)}, rank-{args.rank} LoRA params={params:,}")
        for name, tensor in items:
            out_features, in_features = map(int, tensor.shape)
            print(f"  {in_features:5d}->{out_features:5d}  {FLOW_PREFIX}{name}")

    print("\n" + "-" * 100)
    print(f"Transformer only       : {sum(lora_numel(t,args.rank) for _,t in groups['transformer']):,}")
    print(f"Transformer + enc_vlm : {sum(lora_numel(t,args.rank) for g in ('transformer','enc_vlm') for _,t in groups[g]):,}")
    print(f"Transformer + enc_wm  : {sum(lora_numel(t,args.rank) for g in ('transformer','enc_wm') for _,t in groups[g]):,}")
    print(f"Transformer + output  : {sum(lora_numel(t,args.rank) for g in ('transformer','output') for _,t in groups[g]):,}")
    print(f"All interfaces         : {total:,}")
    print("=" * 100)


if __name__ == "__main__":
    main()
