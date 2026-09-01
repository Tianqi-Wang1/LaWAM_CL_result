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
            if isinstance(nested, dict) and nested and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--adapter-output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    state = load_state_dict(args.checkpoint)
    a_keys = sorted(k for k,v in state.items() if torch.is_tensor(v) and k.endswith(".lora_A"))
    if not a_keys:
        raise RuntimeError("No LoRA tensors found")

    canonical_a = [k for k in a_keys if k.startswith(CANONICAL_PREFIX)]
    if not canonical_a:
        raise RuntimeError("No canonical policy_backend.flow LoRA tensors found")

    adapter = {}
    merged = []
    linear_count = 0
    category_count = 0

    for a_key in a_keys:
        prefix = a_key[:-len(".lora_A")]
        b_key = prefix + ".lora_B"
        if b_key not in state:
            raise RuntimeError(f"Missing B for {a_key}")
        A, B = state[a_key], state[b_key]
        if A.ndim != 2 or B.ndim != 2:
            raise RuntimeError(f"A/B must be 2D for {prefix}")
        rank = int(A.shape[0])
        if int(B.shape[1]) != rank:
            raise RuntimeError(f"Rank mismatch for {prefix}")
        scale = float(args.alpha) / float(rank)
        delta_out_in = (B.float() @ A.float()) * scale  # [out,in]

        w_key = prefix + ".weight"
        W_key = prefix + ".W"
        if w_key in state:
            W = state[w_key]
            if tuple(W.shape) != tuple(delta_out_in.shape):
                raise RuntimeError(f"Linear shape mismatch {prefix}: W={tuple(W.shape)}, delta={tuple(delta_out_in.shape)}")
            state[w_key] = (W.float() + delta_out_in).to(dtype=W.dtype)
            linear_count += 1
        elif W_key in state:
            W = state[W_key]
            if W.ndim != 3:
                raise RuntimeError(f"Category W must be 3D for {prefix}: {tuple(W.shape)}")
            delta_in_out = delta_out_in.t()  # [in,out]
            if tuple(W.shape[1:]) != tuple(delta_in_out.shape):
                raise RuntimeError(f"Category shape mismatch {prefix}: W={tuple(W.shape)}, delta={tuple(delta_in_out.shape)}")
            state[W_key] = (W.float() + delta_in_out.unsqueeze(0)).to(dtype=W.dtype)
            category_count += 1
        else:
            raise RuntimeError(f"No base .weight or .W found for {prefix}")

        if a_key.startswith(CANONICAL_PREFIX):
            adapter[a_key] = A.detach().clone()
            adapter[b_key] = B.detach().clone()
        merged.append(prefix)

    for k in list(state):
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            del state[k]
    leftovers = [k for k in state if ".lora_" in k]
    if leftovers:
        raise RuntimeError(f"LoRA keys remain after merge: {leftovers[:20]}")

    ranks = sorted({int(v.shape[0]) for k,v in adapter.items() if k.endswith(".lora_A")})
    payload = {
        "format": "lawam_flow_interface_lora_v1",
        "alpha": float(args.alpha),
        "rank": ranks[0] if len(ranks)==1 else ranks,
        "num_target_modules": len(canonical_a),
        "num_adapter_tensors": len(adapter),
        "num_adapter_params": sum(v.numel() for v in adapter.values()),
        "state_dict": adapter,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    torch.save(payload, args.adapter_output)

    print("[flow-interface-lora-merge] OK")
    print(f"  canonical targets : {len(canonical_a)}")
    print(f"  merged linear     : {linear_count}")
    print(f"  merged category   : {category_count}")
    print(f"  output            : {args.output}")
    print(f"  adapter           : {args.adapter_output}")


if __name__ == "__main__":
    main()
