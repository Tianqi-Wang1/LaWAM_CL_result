#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return dict(obj)
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and all(torch.is_tensor(v) for v in inner.values()):
                return dict(inner)
    raise RuntimeError(f"Unsupported checkpoint structure: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    sd = load_state(args.input)
    a_keys = sorted(k for k in sd if k.endswith(".lora_A"))
    if not a_keys:
        raise RuntimeError("No LoRA tensors found")

    merged = 0
    for a_key in a_keys:
        prefix = a_key[: -len(".lora_A")]
        b_key = prefix + ".lora_B"
        w_key = prefix + ".weight"
        if b_key not in sd or w_key not in sd:
            raise RuntimeError(f"Incomplete LoRA module: {prefix}")
        A, B, W = sd[a_key], sd[b_key], sd[w_key]
        if A.ndim != 2 or B.ndim != 2 or W.ndim != 2:
            raise RuntimeError(f"Expected 2-D tensors at {prefix}")
        rank = int(A.shape[0])
        if B.shape[1] != rank or W.shape != (B.shape[0], A.shape[1]):
            raise RuntimeError(
                f"Shape mismatch at {prefix}: A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
            )
        delta = (B.float() @ A.float()) * (float(args.alpha) / float(rank))
        sd[w_key] = (W.float() + delta).to(dtype=W.dtype)
        del sd[a_key]
        del sd[b_key]
        merged += 1

    if any(k.endswith(".lora_A") or k.endswith(".lora_B") for k in sd):
        raise RuntimeError("LoRA tensors remain after merge")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.output)
    print(f"[OK] merged {merged} LoRA modules; output={args.output}")


if __name__ == "__main__":
    main()
