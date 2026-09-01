#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
import torch

CANONICAL_FLOW_PREFIX = "policy_backend.flow."
ALIAS_FLOW_PREFIX = "policy_action_head."

TARGET_SUFFIX_PATTERNS = {
    "attn_q": re.compile(r"transformer_blocks\.(\d+)\.attn1\.to_q\.weight$"),
    "attn_k": re.compile(r"transformer_blocks\.(\d+)\.attn1\.to_k\.weight$"),
    "attn_v": re.compile(r"transformer_blocks\.(\d+)\.attn1\.to_v\.weight$"),
    "attn_o": re.compile(r"transformer_blocks\.(\d+)\.attn1\.to_out\.0\.weight$"),
    "ff_in":  re.compile(r"transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.weight$"),
    "ff_out": re.compile(r"transformer_blocks\.(\d+)\.ff\.net\.2\.weight$"),
}

ATTN_GROUP = {"attn_q", "attn_k", "attn_v", "attn_o"}
FFN_GROUP = {"ff_in", "ff_out"}

def load_state_dict(path: Path) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)
            if isinstance(nested, dict) and nested and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return obj

def human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.3f}M"
    if n >= 1_000:
        return f"{n / 1e3:.3f}K"
    return str(n)

def mib_for_params(n: int, bytes_per_param: int) -> float:
    return n * bytes_per_param / (1024 ** 2)

def suffix_without_prefix(key: str) -> str:
    if key.startswith(CANONICAL_FLOW_PREFIX):
        return key[len(CANONICAL_FLOW_PREFIX):]
    if key.startswith(ALIAS_FLOW_PREFIX):
        return key[len(ALIAS_FLOW_PREFIX):]
    return key

def classify_target(key: str):
    suffix = suffix_without_prefix(key)
    for name, pattern in TARGET_SUFFIX_PATTERNS.items():
        m = pattern.search(suffix)
        if m:
            return name, int(m.group(1))
    return None, None

def lora_params_for_weight(weight: torch.Tensor, rank: int) -> int:
    if weight.ndim != 2:
        raise ValueError("LoRA estimator expects a 2D Linear weight.")
    out_features, in_features = map(int, weight.shape)
    return rank * (in_features + out_features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--show-all-flow-2d", action="store_true")
    args = parser.parse_args()

    sd = load_state_dict(args.checkpoint)
    canonical = {k: v for k, v in sd.items() if torch.is_tensor(v) and k.startswith(CANONICAL_FLOW_PREFIX)}
    alias = {k: v for k, v in sd.items() if torch.is_tensor(v) and k.startswith(ALIAS_FLOW_PREFIX)}

    if not canonical:
        raise RuntimeError("No policy_backend.flow.* tensors found.")

    canonical_numel = sum(v.numel() for v in canonical.values())
    alias_numel = sum(v.numel() for v in alias.values())

    print("=" * 100)
    print("LaWAM Flow / LoRA parameter inspection")
    print("=" * 100)
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print("[1] Flow checkpoint namespaces")
    print(f"  canonical tensors : {len(canonical)}")
    print(f"  canonical params  : {canonical_numel:,} ({human_params(canonical_numel)})")
    print(f"  alias tensors     : {len(alias)}")
    print(f"  alias params      : {alias_numel:,} ({human_params(alias_numel)})")
    print("  NOTE: do NOT add canonical and alias counts together.")

    targets = []
    blocks = defaultdict(set)
    for key, value in canonical.items():
        category, block_idx = classify_target(key)
        if category is None or value.ndim != 2:
            continue
        targets.append((key, value, category, block_idx))
        blocks[block_idx].add(category)

    print()
    print("[2] Recommended first-pass LoRA targets found")
    print(f"  matched linear weights: {len(targets)} across {len(blocks)} blocks")
    for key, value, category, block_idx in sorted(targets, key=lambda x: (x[3], x[2], x[0])):
        out_features, in_features = map(int, value.shape)
        print(f"  block={block_idx:02d} {category:7s} {in_features:5d}->{out_features:5d} {key}")

    expected = ATTN_GROUP | FFN_GROUP
    print()
    print("[3] Per-block completeness")
    for idx in sorted(blocks):
        missing = sorted(expected - blocks[idx])
        print(f"  block {idx:02d}: present={sorted(blocks[idx])}" + (f" MISSING={missing}" if missing else ""))

    attn_targets = [x for x in targets if x[2] in ATTN_GROUP]
    attn_ffn_targets = [x for x in targets if x[2] in (ATTN_GROUP | FFN_GROUP)]

    def estimate_group(name, group):
        print()
        print(f"[4] LoRA estimate: {name}")
        base_target_params = sum(v.numel() for _, v, _, _ in group)
        print(f"  frozen base Linear weights covered: {base_target_params:,} ({human_params(base_target_params)})")
        for rank in args.ranks:
            lora_numel = sum(lora_params_for_weight(v, rank) for _, v, _, _ in group)
            pct_flow = 100.0 * lora_numel / canonical_numel
            print(
                f"  r={rank:<3d} LoRA={lora_numel:>12,} ({human_params(lora_numel):>8s}) "
                f"= {pct_flow:6.3f}% of full Flow | "
                f"BF16={mib_for_params(lora_numel, 2):7.2f} MiB "
                f"FP32={mib_for_params(lora_numel, 4):7.2f} MiB"
            )

    estimate_group("Attention Q/K/V/O only", attn_targets)
    estimate_group("Attention Q/K/V/O + FFN", attn_ffn_targets)

    print()
    print("[5] Full Flow storage reference")
    print(f"  BF16 full Flow: {mib_for_params(canonical_numel, 2):.2f} MiB")
    print(f"  FP32 full Flow: {mib_for_params(canonical_numel, 4):.2f} MiB")

    if args.show_all_flow_2d:
        print()
        print("[6] ALL canonical Flow 2D tensors")
        for key, value in sorted(canonical.items()):
            if value.ndim == 2:
                out_features, in_features = map(int, value.shape)
                category, _ = classify_target(key)
                print(f"  {in_features:5d}->{out_features:5d} target={(category or '-'):7s} {key}")

if __name__ == "__main__":
    main()