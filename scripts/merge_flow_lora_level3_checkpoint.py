#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


CANONICAL_PREFIX = "policy_backend.flow."
ALIAS_PREFIX = "policy_action_head."


def load_state_dict(path: Path) -> dict:
    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        try:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
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
            "Merge LaWAM Flow-LoRA Level-3 adapters into a standard "
            "non-LoRA checkpoint."
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
        raise RuntimeError("No .lora_A tensors found.")

    canonical_a_keys = [
        k for k in lora_a_keys
        if k.startswith(CANONICAL_PREFIX)
    ]
    alias_a_keys = [
        k for k in lora_a_keys
        if k.startswith(ALIAS_PREFIX)
    ]

    if not canonical_a_keys:
        raise RuntimeError(
            "No canonical policy_backend.flow.* LoRA tensors found."
        )

    adapter_state = {}
    standard_count = 0
    category_count = 0
    merged_modules = []

    for a_key in lora_a_keys:
        prefix = a_key[:-len(".lora_A")]
        b_key = prefix + ".lora_B"

        if b_key not in state:
            raise RuntimeError(
                f"Missing LoRA B tensor for {a_key}: {b_key}"
            )

        A = state[a_key]
        B = state[b_key]

        # ---------------------------------------------------------------------
        # Ordinary nn.Linear:
        #   A [r, in], B [out, r], weight [out, in]
        # ---------------------------------------------------------------------
        weight_key = prefix + ".weight"
        category_weight_key = prefix + ".W"

        if weight_key in state:
            W = state[weight_key]

            if A.ndim != 2 or B.ndim != 2 or W.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D standard LoRA tensors for {prefix}: "
                    f"A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
                )

            rank = int(A.shape[0])

            if int(B.shape[1]) != rank:
                raise RuntimeError(
                    f"Rank mismatch for {prefix}: "
                    f"A={tuple(A.shape)}, B={tuple(B.shape)}"
                )

            expected_w = (int(B.shape[0]), int(A.shape[1]))
            if tuple(W.shape) != expected_w:
                raise RuntimeError(
                    f"Standard weight mismatch for {prefix}: "
                    f"W={tuple(W.shape)}, expected={expected_w}"
                )

            scaling = float(args.alpha) / float(rank)
            delta = (B.float() @ A.float()) * scaling
            state[weight_key] = (
                W.float() + delta
            ).to(dtype=W.dtype)

            standard_count += 1

        # ---------------------------------------------------------------------
        # CategorySpecificLinear:
        #   W [C, in, out]
        #   A [in, r], B [r, out]
        #
        # Level-3 uses a task residual shared across categories.
        # For current LIBERO/Franka inference only one category is active.
        # ---------------------------------------------------------------------
        elif category_weight_key in state:
            W = state[category_weight_key]

            if A.ndim != 2 or B.ndim != 2 or W.ndim != 3:
                raise RuntimeError(
                    f"Expected category LoRA A/B 2D and W 3D for {prefix}: "
                    f"A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
                )

            rank = int(A.shape[1])

            if int(B.shape[0]) != rank:
                raise RuntimeError(
                    f"Category rank mismatch for {prefix}: "
                    f"A={tuple(A.shape)}, B={tuple(B.shape)}"
                )

            expected_io = (int(A.shape[0]), int(B.shape[1]))
            if tuple(W.shape[1:]) != expected_io:
                raise RuntimeError(
                    f"Category weight mismatch for {prefix}: "
                    f"W={tuple(W.shape)}, expected_io={expected_io}"
                )

            scaling = float(args.alpha) / float(rank)
            delta = (A.float() @ B.float()) * scaling  # [in, out]

            state[category_weight_key] = (
                W.float() + delta.unsqueeze(0)
            ).to(dtype=W.dtype)

            category_count += 1

        else:
            raise RuntimeError(
                f"No base `.weight` or `.W` found for LoRA module {prefix}"
            )

        merged_modules.append(prefix)

        if a_key.startswith(CANONICAL_PREFIX):
            adapter_state[a_key] = A.detach().clone()
            adapter_state[b_key] = B.detach().clone()

    # Remove all LoRA tensors to recover standard LaWAM architecture.
    for key in list(state.keys()):
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            del state[key]

    remaining = [
        key for key in state
        if ".lora_" in key
    ]
    if remaining:
        raise RuntimeError(
            f"Merged checkpoint still contains LoRA keys: {remaining[:20]}"
        )

    adapter_numel = sum(v.numel() for v in adapter_state.values())

    payload = {
        "format": "lawam_flow_lora_level3_v1",
        "alpha": float(args.alpha),
        "canonical_prefix": CANONICAL_PREFIX,
        "category_lora_mode": "shared_task_residual",
        "num_target_modules": len(canonical_a_keys),
        "num_adapter_tensors": len(adapter_state),
        "num_adapter_params": adapter_numel,
        "state_dict": adapter_state,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, args.output)
    torch.save(payload, args.adapter_output)

    print("[flow-lora-level3-merge] OK")
    print(f"  input                : {args.checkpoint}")
    print(f"  merged checkpoint    : {args.output}")
    print(f"  adapter checkpoint   : {args.adapter_output}")
    print(f"  canonical targets    : {len(canonical_a_keys)}")
    print(f"  alias targets        : {len(alias_a_keys)}")
    print(f"  standard targets     : {standard_count // 2 if standard_count else 0} canonical-equivalent")
    print(f"  category targets     : {category_count // 2 if category_count else 0} canonical-equivalent")
    print(f"  adapter tensors      : {len(adapter_state)}")
    print(f"  adapter params       : {adapter_numel:,}")
    print(f"  merged namespace mods: {len(merged_modules)}")


if __name__ == "__main__":
    main()
