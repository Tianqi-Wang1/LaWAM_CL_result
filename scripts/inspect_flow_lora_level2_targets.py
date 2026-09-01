#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


FLOW_PREFIX = "policy_backend.flow."

TRANSFORMER_PATTERNS = (
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_q\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_k\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_v\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_out\.0\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.ff\.net\.0\.proj\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.ff\.net\.2\.weight$"),
)


def load(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(path, map_location="cpu")


def lora_params(weight: torch.Tensor, rank: int) -> int:
    out_features, in_features = map(int, weight.shape)
    return rank * (in_features + out_features)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    sd = load(args.checkpoint)

    groups = {
        "transformer": [],
        "enc_vlm": [],
        "output": [],
    }

    for key, value in sd.items():
        if (
            not torch.is_tensor(value)
            or not key.startswith(FLOW_PREFIX)
            or value.ndim != 2
        ):
            continue

        suffix = key[len(FLOW_PREFIX):]

        if any(p.match(suffix) for p in TRANSFORMER_PATTERNS):
            groups["transformer"].append((suffix, value))
        elif (
            suffix.startswith("enc_vlm.")
            and suffix.endswith(".weight")
        ):
            groups["enc_vlm"].append((suffix, value))
        elif suffix in (
            "DiT.proj_out_1.weight",
            "DiT.proj_out_2.weight",
        ):
            groups["output"].append((suffix, value))

    print("=" * 96)
    print(f"LaWAM Flow-LoRA Level-2 target inspection (rank={args.rank})")
    print("=" * 96)

    counts = {}
    for group in ("transformer", "enc_vlm", "output"):
        numel = sum(
            lora_params(w, args.rank)
            for _, w in groups[group]
        )
        counts[group] = numel
        print(
            f"{group:12s}: "
            f"modules={len(groups[group]):3d}, "
            f"LoRA params={numel:,}"
        )
        for name, w in groups[group]:
            o, i = map(int, w.shape)
            print(f"  {i:5d} -> {o:5d}  {name}")

    print("-" * 96)
    variants = (
        ("T only", ("transformer",)),
        ("T + enc_vlm", ("transformer", "enc_vlm")),
        ("T + output", ("transformer", "output")),
        (
            "T + enc_vlm + output",
            ("transformer", "enc_vlm", "output"),
        ),
    )

    for name, gs in variants:
        n = sum(counts[g] for g in gs)
        print(f"{name:24s}: {n:,} params")

    if len(groups["transformer"]) != 96:
        raise RuntimeError(
            f"Expected 96 Transformer target weights, "
            f"found {len(groups['transformer'])}"
        )
    if len(groups["enc_vlm"]) != 1:
        raise RuntimeError(
            f"Expected 1 enc_vlm Linear weight in current model, "
            f"found {len(groups['enc_vlm'])}"
        )
    if len(groups["output"]) != 2:
        raise RuntimeError(
            f"Expected 2 output Linear weights, "
            f"found {len(groups['output'])}"
        )


if __name__ == "__main__":
    main()
