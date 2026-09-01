#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


PREFIX = "policy_backend.flow."

BASE_PATTERNS = (
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


def lora_numel(weight, rank):
    out_f, in_f = map(int, weight.shape)
    return rank * (in_f + out_f)


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
        "adanorm": [],
        "timestep": [],
    }

    for key, value in sd.items():
        if (
            not torch.is_tensor(value)
            or not key.startswith(PREFIX)
            or value.ndim != 2
        ):
            continue

        suffix = key[len(PREFIX):]

        if any(p.match(suffix) for p in BASE_PATTERNS):
            groups["transformer"].append((suffix, value))
        elif suffix == "enc_vlm.net.0.weight":
            groups["enc_vlm"].append((suffix, value))
        elif suffix in (
            "DiT.proj_out_1.weight",
            "DiT.proj_out_2.weight",
        ):
            groups["output"].append((suffix, value))
        elif re.match(
            r"^DiT\.transformer_blocks\.\d+\.norm1\..*linear.*\.weight$",
            suffix,
        ) or re.match(
            r"^DiT\.transformer_blocks\.\d+\.norm1\.linear\.weight$",
            suffix,
        ):
            groups["adanorm"].append((suffix, value))
        elif (
            suffix.startswith("DiT.timestep_encoder.")
            and suffix.endswith(".weight")
        ):
            groups["timestep"].append((suffix, value))

    print("=" * 100)
    print(
        f"LaWAM T9 Conditioning-Complete target inspection "
        f"(rank={args.rank})"
    )
    print("=" * 100)

    counts = {}
    for group in (
        "transformer",
        "enc_vlm",
        "output",
        "adanorm",
        "timestep",
    ):
        n = sum(
            lora_numel(w, args.rank)
            for _, w in groups[group]
        )
        counts[group] = n
        print(
            f"{group:12s}: modules={len(groups[group]):3d}, "
            f"LoRA params={n:,}"
        )
        for name, w in groups[group]:
            out_f, in_f = map(int, w.shape)
            print(
                f"  {in_f:5d} -> {out_f:5d}  {name}"
            )

    ref = (
        counts["transformer"]
        + counts["enc_vlm"]
        + counts["output"]
    )
    adanorm = ref + counts["adanorm"]
    timestep = ref + counts["timestep"]
    both = ref + counts["adanorm"] + counts["timestep"]

    print("-" * 100)
    print(f"Fresh Level-2 reference       : {ref:,}")
    print(f"Level-2 + AdaNorm             : {adanorm:,}")
    print(f"Level-2 + timestep            : {timestep:,}")
    print(f"Conditioning-complete         : {both:,}")
    print("=" * 100)

    expected_counts = {
        "transformer": 96,
        "enc_vlm": 1,
        "output": 2,
        "adanorm": 16,
        "timestep": 2,
    }

    for group, expected in expected_counts.items():
        if len(groups[group]) != expected:
            raise RuntimeError(
                f"{group}: expected {expected} modules in current LaWAM "
                f"checkpoint, found {len(groups[group])}"
            )

    if args.rank == 8:
        expected_params = {
            "transformer": 2_326_528,
            "enc_vlm": 22_528,
            "output": 40_960,
            "adanorm": 393_216,
            "timestep": 26_624,
        }
        for group, expected in expected_params.items():
            if counts[group] != expected:
                raise RuntimeError(
                    f"{group}: expected r8 params={expected:,}, "
                    f"found={counts[group]:,}"
                )


if __name__ == "__main__":
    main()
