#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def is_upstream_delta_key(key: str) -> bool:
    if key in {
        "policy_backend.routing_v2_act_query_delta",
        "policy_backend.routing_v2_flow_query_delta",
    }:
        return True
    if key.startswith("policy_backend.vlm.") and (
        key.endswith(".lora_A") or key.endswith(".lora_B")
    ):
        return True
    if key.startswith("policy_backend.vlm_to_lam.") and (
        key.endswith(".lora_A") or key.endswith(".lora_B")
    ):
        return True
    if key.startswith("policy_backend.lam.decoder.") and (
        key.endswith(".lora_A") or key.endswith(".lora_B")
    ):
        return True
    return False


def group_for(key: str) -> str:
    if key.startswith("policy_backend.vlm."):
        return "vlm_text_lora"
    if key.startswith("policy_backend.vlm_to_lam."):
        return "qformer_lora"
    if key.startswith("policy_backend.lam.decoder."):
        return "lawm_lora"
    return "query_delta"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    state = _load(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint is not a state dict: {ckpt}")

    delta = {
        k: v.detach().cpu().clone()
        for k, v in state.items()
        if torch.is_tensor(v) and is_upstream_delta_key(k)
    }
    if not delta:
        raise RuntimeError(f"No Routing-V2 upstream task delta tensors found in {ckpt}")

    groups = {"vlm_text_lora": 0, "qformer_lora": 0, "lawm_lora": 0, "query_delta": 0}
    counts = {k: 0 for k in groups}
    for k, v in delta.items():
        g = group_for(k)
        groups[g] += int(v.numel())
        counts[g] += 1

    expected_nonzero = [g for g, n in groups.items() if n <= 0]
    if expected_nonzero:
        raise RuntimeError(f"Missing expected Routing-V2 upstream groups: {expected_nonzero}; groups={groups}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, out)
    meta = {
        "checkpoint": str(ckpt),
        "output": str(out),
        "tensor_count": len(delta),
        "groups_numel": groups,
        "groups_tensor_count": counts,
        "total_numel": sum(int(v.numel()) for v in delta.values()),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[RoutingV2][UPSTREAM-DELTA] extracted")
    print(f"  checkpoint : {ckpt}")
    print(f"  output     : {out}")
    for g in groups:
        print(f"  {g:16s}: tensors={counts[g]:4d} params={groups[g]:,}")
    print(f"  {'total':16s}: tensors={len(delta):4d} params={meta['total_numel']:,}")


if __name__ == "__main__":
    main()
