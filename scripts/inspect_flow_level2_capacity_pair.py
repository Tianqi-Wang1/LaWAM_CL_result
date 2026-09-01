#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


PREFIX = "policy_backend.flow."

TRANSFORMER_PATTERNS = (
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_q\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_k\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_v\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.attn1\.to_out\.0\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.ff\.net\.0\.proj\.weight$"),
    re.compile(r"^DiT\.transformer_blocks\.\d+\.ff\.net\.2\.weight$"),
)


def load(path):
    try:
        return torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


def lora_numel(W, r):
    out_f, in_f = map(int, W.shape)
    return r * (in_f + out_f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    args = ap.parse_args()

    sd = load(args.checkpoint)

    transformer = []
    interfaces = []

    for k, v in sd.items():
        if (
            not torch.is_tensor(v)
            or not k.startswith(PREFIX)
            or v.ndim != 2
        ):
            continue

        s = k[len(PREFIX):]

        if any(p.match(s) for p in TRANSFORMER_PATTERNS):
            transformer.append((s, v))
        elif (
            s == "enc_vlm.net.0.weight"
            or s in (
                "DiT.proj_out_1.weight",
                "DiT.proj_out_2.weight",
            )
        ):
            interfaces.append((s, v))

    if len(transformer) != 96:
        raise RuntimeError(
            f"Expected 96 Transformer targets, got {len(transformer)}"
        )
    if len(interfaces) != 3:
        raise RuntimeError(
            f"Expected 3 interface weights, got {len(interfaces)}"
        )

    t_r8 = sum(lora_numel(w, 8) for _, w in transformer)
    all_r32 = (
        sum(lora_numel(w, 32) for _, w in transformer)
        + sum(lora_numel(w, 32) for _, w in interfaces)
    )
    dense_interfaces = sum(w.numel() for _, w in interfaces)
    mixed = t_r8 + dense_interfaces

    print("=" * 92)
    print("T9 Level-2 capacity pair inspection")
    print("=" * 92)
    print(f"Transformer targets          : {len(transformer)}")
    print(f"Interface targets            : {len(interfaces)}")
    print(f"Transformer LoRA r8          : {t_r8:,}")
    print(f"Broad Level-2 LoRA r32       : {all_r32:,}")
    print(f"Dense interface delta        : {dense_interfaces:,}")
    print(f"T-r8 + dense interfaces      : {mixed:,}")
    print("-" * 92)

    for name, W in interfaces:
        out_f, in_f = map(int, W.shape)
        print(
            f"{name:32s} {in_f:5d}->{out_f:5d} "
            f"dense={W.numel():,} "
            f"lora-r32={lora_numel(W,32):,}"
        )

    expected = {
        "t_r8": 2_326_528,
        "all_r32": 9_560_064,
        "dense": 4_718_592,
        "mixed": 7_045_120,
    }
    actual = {
        "t_r8": t_r8,
        "all_r32": all_r32,
        "dense": dense_interfaces,
        "mixed": mixed,
    }

    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(
                f"{key}: expected {value:,}, got {actual[key]:,}"
            )


if __name__ == "__main__":
    main()
