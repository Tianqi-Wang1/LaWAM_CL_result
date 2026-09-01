#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


CANONICAL_PREFIX = "policy_backend.flow."
ALIAS_PREFIX = "policy_action_head."


def load_state(path: Path):
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
        raise RuntimeError(f"Unsupported checkpoint: {path}")

    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--adapter-output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    state = load_state(args.checkpoint)

    adapter_state = {}
    merged_prefixes = set()

    # -------------------------------------------------------------------------
    # Merge LoRA.
    # -------------------------------------------------------------------------
    lora_a_keys = sorted(
        k for k, v in state.items()
        if torch.is_tensor(v) and k.endswith(".lora_A")
    )

    for a_key in lora_a_keys:
        prefix = a_key[:-len(".lora_A")]
        b_key = prefix + ".lora_B"
        w_key = prefix + ".weight"

        if b_key not in state:
            raise RuntimeError(f"Missing {b_key}")
        if w_key not in state:
            raise RuntimeError(
                f"Pair package expects ordinary Linear target; "
                f"missing {w_key}"
            )

        A = state[a_key]
        B = state[b_key]
        W = state[w_key]

        if A.ndim != 2 or B.ndim != 2 or W.ndim != 2:
            raise RuntimeError(
                f"Invalid LoRA shapes for {prefix}: "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
            )

        rank = int(A.shape[0])
        if int(B.shape[1]) != rank:
            raise RuntimeError(f"Rank mismatch for {prefix}")

        scale = float(args.alpha) / float(rank)
        delta = (B.float() @ A.float()) * scale

        if tuple(delta.shape) != tuple(W.shape):
            raise RuntimeError(
                f"Delta shape mismatch for {prefix}: "
                f"{tuple(delta.shape)} != {tuple(W.shape)}"
            )

        state[w_key] = (W.float() + delta).to(dtype=W.dtype)
        merged_prefixes.add(prefix)

        if a_key.startswith(CANONICAL_PREFIX):
            adapter_state[a_key] = A.detach().clone()
            adapter_state[b_key] = B.detach().clone()

    # -------------------------------------------------------------------------
    # Merge full-rank interface deltas.
    # -------------------------------------------------------------------------
    dense_keys = sorted(
        k for k, v in state.items()
        if torch.is_tensor(v) and k.endswith(".delta_weight")
    )

    for d_key in dense_keys:
        prefix = d_key[:-len(".delta_weight")]
        w_key = prefix + ".weight"

        if w_key not in state:
            raise RuntimeError(
                f"Missing Base weight for dense delta: {w_key}"
            )

        D = state[d_key]
        W = state[w_key]

        if tuple(D.shape) != tuple(W.shape):
            raise RuntimeError(
                f"Dense delta shape mismatch for {prefix}: "
                f"D={tuple(D.shape)}, W={tuple(W.shape)}"
            )

        state[w_key] = (W.float() + D.float()).to(dtype=W.dtype)
        merged_prefixes.add(prefix)

        if d_key.startswith(CANONICAL_PREFIX):
            adapter_state[d_key] = D.detach().clone()

    if not lora_a_keys and not dense_keys:
        raise RuntimeError("No adapter tensors found.")

    # Remove all task adapter tensors for a standard LaWAM checkpoint.
    for key in list(state.keys()):
        if (
            key.endswith(".lora_A")
            or key.endswith(".lora_B")
            or key.endswith(".delta_weight")
        ):
            del state[key]

    remaining = [
        k for k in state
        if (
            ".lora_" in k
            or k.endswith(".delta_weight")
        )
    ]
    if remaining:
        raise RuntimeError(
            f"Merged checkpoint still has adapter keys: {remaining[:20]}"
        )

    payload = {
        "format": "lawam_flow_level2_capacity_pair_v1",
        "alpha": float(args.alpha),
        "canonical_prefix": CANONICAL_PREFIX,
        "num_adapter_tensors": len(adapter_state),
        "num_adapter_params": sum(
            v.numel() for v in adapter_state.values()
        ),
        "state_dict": adapter_state,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, args.output)
    torch.save(payload, args.adapter_output)

    print("[flow-level2-pair-merge] OK")
    print(f"  LoRA namespace targets : {len(lora_a_keys)}")
    print(f"  dense namespace targets: {len(dense_keys)}")
    print(f"  merged prefixes        : {len(merged_prefixes)}")
    print(f"  adapter tensors        : {len(adapter_state)}")
    print(
        f"  adapter params         : "
        f"{sum(v.numel() for v in adapter_state.values()):,}"
    )
    print(f"  merged checkpoint      : {args.output}")


if __name__ == "__main__":
    main()
