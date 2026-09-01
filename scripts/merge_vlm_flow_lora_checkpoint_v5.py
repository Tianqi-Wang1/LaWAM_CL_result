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
    ap.add_argument("--alpha", type=float, default=8.0)
    args = ap.parse_args()

    sd = load_state(args.input)
    a_keys = sorted(k for k in sd if k.endswith(".lora_A"))
    b_keys = sorted(k for k in sd if k.endswith(".lora_B"))
    if not a_keys or len(a_keys) != len(b_keys):
        raise RuntimeError(f"Invalid LoRA key counts: A={len(a_keys)}, B={len(b_keys)}")

    merged = 0
    flow_merged = 0
    vlm_merged = 0
    for a_key in a_keys:
        prefix = a_key[: -len(".lora_A")]
        b_key = prefix + ".lora_B"
        w_key = prefix + ".weight"
        if b_key not in sd:
            raise RuntimeError(f"Missing {b_key}")
        if w_key not in sd:
            raise RuntimeError(f"Missing Base Linear weight for LoRA module: {w_key}")
        A = sd[a_key]
        B = sd[b_key]
        W = sd[w_key]
        if A.ndim != 2 or B.ndim != 2 or W.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D Linear tensors for {prefix}: A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
            )
        rank = int(A.shape[0])
        if B.shape[1] != rank or W.shape != (B.shape[0], A.shape[1]):
            raise RuntimeError(
                f"LoRA shape mismatch for {prefix}: A={tuple(A.shape)}, B={tuple(B.shape)}, W={tuple(W.shape)}"
            )
        scaling = float(args.alpha) / float(rank)
        delta = (B.float() @ A.float()) * scaling
        sd[w_key] = (W.float() + delta).to(dtype=W.dtype)
        del sd[a_key]
        del sd[b_key]
        merged += 1
        if prefix.startswith("policy_backend.vlm."):
            vlm_merged += 1
        if prefix.startswith("policy_backend.flow.") or prefix.startswith("policy_action_head."):
            flow_merged += 1

    if any(k.endswith(".lora_A") or k.endswith(".lora_B") for k in sd):
        raise RuntimeError("LoRA keys remain after merge")
    if vlm_merged <= 0 or flow_merged <= 0:
        raise RuntimeError(f"Expected both VLM and Flow LoRA modules; got vlm={vlm_merged}, flow={flow_merged}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, args.output)
    print(
        f"[OK] merged {merged} LoRA modules into Base weights: "
        f"vlm={vlm_merged}, flow_or_alias={flow_merged}; output={args.output}"
    )


if __name__ == "__main__":
    main()
