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


def standard_lora_params(weight: torch.Tensor, rank: int) -> int:
    out_features, in_features = map(int, weight.shape)
    return rank * (in_features + out_features)


def category_shared_lora_params(W: torch.Tensor, rank: int) -> int:
    if W.ndim != 3:
        raise ValueError(tuple(W.shape))
    in_features = int(W.shape[1])
    out_features = int(W.shape[2])
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
        "action_encoder": [],
        "action_decoder": [],
    }

    for key, value in sd.items():
        if (
            not torch.is_tensor(value)
            or not key.startswith(FLOW_PREFIX)
        ):
            continue

        suffix = key[len(FLOW_PREFIX):]

        if value.ndim == 2:
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

        elif value.ndim == 3:
            if suffix in (
                "action_encoder.W1.W",
                "action_encoder.W2.W",
            ):
                groups["action_encoder"].append((suffix, value))
            elif suffix in (
                "action_decoder.layer1.W",
                "action_decoder.layer2.W",
            ):
                groups["action_decoder"].append((suffix, value))

    counts = {}

    print("=" * 100)
    print(
        f"LaWAM Flow-LoRA Level-3 target inspection "
        f"(rank={args.rank}, category mode=shared task residual)"
    )
    print("=" * 100)

    for group in (
        "transformer",
        "enc_vlm",
        "output",
        "action_encoder",
        "action_decoder",
    ):
        if group in ("action_encoder", "action_decoder"):
            numel = sum(
                category_shared_lora_params(w, args.rank)
                for _, w in groups[group]
            )
        else:
            numel = sum(
                standard_lora_params(w, args.rank)
                for _, w in groups[group]
            )

        counts[group] = numel

        print(
            f"{group:16s}: modules={len(groups[group]):3d}, "
            f"LoRA params={numel:,}"
        )

        for name, w in groups[group]:
            if w.ndim == 2:
                out_f, in_f = map(int, w.shape)
                print(
                    f"  standard       {in_f:5d} -> {out_f:5d}  {name}"
                )
            else:
                c, in_f, out_f = map(int, w.shape)
                print(
                    f"  category[C={c:2d}] {in_f:5d} -> {out_f:5d}  {name}"
                )

    level2 = (
        counts["transformer"]
        + counts["enc_vlm"]
        + counts["output"]
    )
    l3_enc = level2 + counts["action_encoder"]
    l3_dec = level2 + counts["action_decoder"]
    l3_full = (
        level2
        + counts["action_encoder"]
        + counts["action_decoder"]
    )

    print("-" * 100)
    print(f"Level-2 reference             : {level2:,}")
    print(f"Level-2 + action encoder      : {l3_enc:,}")
    print(f"Level-2 + action decoder      : {l3_dec:,}")
    print(f"Full Level-3                  : {l3_full:,}")
    print("=" * 100)

    expected_modules = {
        "transformer": 96,
        "enc_vlm": 1,
        "output": 2,
        "action_encoder": 2,
        "action_decoder": 2,
    }

    for group, expected in expected_modules.items():
        if len(groups[group]) != expected:
            raise RuntimeError(
                f"{group}: expected {expected} target tensors/modules, "
                f"found {len(groups[group])}"
            )

    # Current expected r8 counts from the real checkpoint architecture.
    if args.rank == 8:
        expected = {
            "transformer": 2_326_528,
            "enc_vlm": 22_528,
            "output": 40_960,
            "action_encoder": 24_832,
            "action_decoder": 24_832,
        }
        for group, value in expected.items():
            if counts[group] != value:
                raise RuntimeError(
                    f"{group}: expected r8 params={value:,}, "
                    f"found={counts[group]:,}"
                )


if __name__ == "__main__":
    main()
