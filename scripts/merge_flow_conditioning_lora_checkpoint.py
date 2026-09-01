#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


CANONICAL_PREFIX = "policy_backend.flow."
ALIAS_PREFIX = "policy_action_head."


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
            if (
                isinstance(nested, dict)
                and nested
                and any(torch.is_tensor(v) for v in nested.values())
            ):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge trained LaWAM Flow-LoRA tensors into the original Linear weights "
            "and save a standard architecture-compatible checkpoint."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    state = load_state_dict(args.checkpoint)

    lora_a_keys = sorted(
        key
        for key, value in state.items()
        if torch.is_tensor(value) and key.endswith(".lora_A")
    )
    if not lora_a_keys:
        raise RuntimeError("No .lora_A tensors found in checkpoint.")

    canonical_a_keys = [k for k in lora_a_keys if k.startswith(CANONICAL_PREFIX)]
    alias_a_keys = [k for k in lora_a_keys if k.startswith(ALIAS_PREFIX)]

    if not canonical_a_keys:
        raise RuntimeError("No canonical policy_backend.flow.* LoRA tensors found.")

    adapter_state = {}
    merged_modules = []

    # Merge both canonical and alias namespaces so the standard LaWAM checkpoint
    # remains internally consistent with its duplicated action-head registration.
    for a_key in lora_a_keys:
        prefix = a_key[: -len(".lora_A")]
        b_key = prefix + ".lora_B"
        w_key = prefix + ".weight"

        if b_key not in state:
            raise RuntimeError(f"Missing LoRA B tensor for {a_key}: {b_key}")
        if w_key not in state:
            raise RuntimeError(f"Missing base Linear weight for {a_key}: {w_key}")

        A = state[a_key]
        B = state[b_key]
        W = state[w_key]

        if A.ndim != 2 or B.ndim != 2 or W.ndim != 2:
            raise RuntimeError(
                f"Expected 2D A/B/W for {prefix}: "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
            )

        rank = int(A.shape[0])
        if B.shape[1] != rank:
            raise RuntimeError(
                f"Rank mismatch for {prefix}: A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
        if tuple(W.shape) != (int(B.shape[0]), int(A.shape[1])):
            raise RuntimeError(
                f"Weight/LoRA shape mismatch for {prefix}: "
                f"W={tuple(W.shape)}, B@A={(int(B.shape[0]), int(A.shape[1]))}"
            )

        scaling = float(args.alpha) / float(rank)
        delta = (B.float() @ A.float()) * scaling
        state[w_key] = (W.float() + delta).to(dtype=W.dtype)
        merged_modules.append(prefix)

        if a_key.startswith(CANONICAL_PREFIX):
            adapter_state[a_key] = A.detach().clone()
            adapter_state[b_key] = B.detach().clone()

    # Remove LoRA tensors from the merged checkpoint.  The resulting checkpoint
    # can be loaded by the original non-LoRA LaWAM inference architecture.
    remove_keys = [
        key
        for key in list(state.keys())
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    ]
    for key in remove_keys:
        del state[key]

    remaining_lora = [
        key for key in state if ".lora_" in key
    ]
    if remaining_lora:
        raise RuntimeError(
            f"Merged checkpoint still contains LoRA keys: {remaining_lora[:20]}"
        )

    adapter_numel = sum(v.numel() for v in adapter_state.values())
    rank_values = sorted({int(v.shape[0]) for k, v in adapter_state.items() if k.endswith(".lora_A")})

    payload = {
        "format": "lawam_flow_lora_v1",
        "alpha": float(args.alpha),
        "rank": rank_values[0] if len(rank_values) == 1 else rank_values,
        "canonical_prefix": CANONICAL_PREFIX,
        "num_target_modules": len(canonical_a_keys),
        "num_adapter_tensors": len(adapter_state),
        "num_adapter_params": adapter_numel,
        "state_dict": adapter_state,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, args.output)
    torch.save(payload, args.adapter_output)

    print("[flow-lora-merge] OK")
    print(f"  input               : {args.checkpoint}")
    print(f"  merged checkpoint   : {args.output}")
    print(f"  adapter checkpoint  : {args.adapter_output}")
    print(f"  canonical targets   : {len(canonical_a_keys)}")
    print(f"  alias targets       : {len(alias_a_keys)}")
    print(f"  adapter tensors     : {len(adapter_state)}")
    print(f"  adapter params      : {adapter_numel:,}")
    print(f"  merged namespace mods: {len(merged_modules)}")


if __name__ == "__main__":
    main()
