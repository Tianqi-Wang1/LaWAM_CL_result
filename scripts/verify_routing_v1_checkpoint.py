#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected state_dict at {path}, got {type(obj).__name__}")
    return obj


def get_cfg(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def canonical_head_items(sd: Dict[str, torch.Tensor]):
    prefix = "policy_backend.flow.expert_latent_head."
    return {k: v for k, v in sd.items() if k.startswith(prefix)}


def assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
        raise RuntimeError(f"Frozen tensor changed unexpectedly: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--mode", choices=["base", "b1", "b2"], required=True)
    args = ap.parse_args()

    sd = load_state(args.checkpoint)
    cfg = get_cfg(args.config)
    am = cfg["framework"]["action_model"]
    flow_cfg = am["flow_cfg"]
    freeze = cfg["trainer"]["freeze"]

    if not bool(flow_cfg.get("enable_expert_latent_head", False)):
        raise RuntimeError("Routing-V1 checkpoint config does not enable expert latent head")
    if not bool(am.get("enable_expert_latent_aux", False)):
        raise RuntimeError("Routing-V1 checkpoint config does not enable expert latent auxiliary loss")

    head = canonical_head_items(sd)
    if not head:
        raise RuntimeError("Checkpoint contains no canonical policy_backend.flow.expert_latent_head tensors")
    if not all(torch.isfinite(v).all() for v in head.values() if torch.is_floating_point(v)):
        raise RuntimeError("Expert latent head contains non-finite values")

    head_params = sum(v.numel() for v in head.values())
    fc2 = head.get("policy_backend.flow.expert_latent_head.fc2.weight")
    if fc2 is None or fc2.ndim != 2:
        raise RuntimeError("Missing expert latent fc2.weight")
    print(
        f"[OK] expert latent head: params={head_params:,}, "
        f"latent_dim={fc2.shape[0]}, input_hidden={head['policy_backend.flow.expert_latent_head.norm.weight'].numel()}"
    )

    if args.mode == "base":
        forbidden = [
            "freeze_vlm_all",
            "freeze_act_query",
            "freeze_flow_action_query",
        ]
        bad = [name for name in forbidden if bool(freeze.get(name, False))]
        if bad:
            raise RuntimeError(f"Base Routing-V1 should not freeze VLM/query interfaces: {bad}")
        if not bool(freeze.get("unfreeze_lam_decoder", False)):
            raise RuntimeError("Base Routing-V1 must keep LaWM/LAM decoder trainable")
        print("[OK] Base protocol: VLM/query interfaces trainable; LaWM decoder trainable.")
        return

    if args.base is None:
        raise RuntimeError("--base is required for B1/B2 verification")
    base = load_state(args.base)

    if not bool(freeze.get("freeze_vlm_all", False)):
        raise RuntimeError("B1/B2 must freeze the shared VLM")
    if not bool(freeze.get("freeze_act_query", False)):
        raise RuntimeError("B1/B2 must freeze act_query")
    if not bool(freeze.get("freeze_flow_action_query", False)):
        raise RuntimeError("B1/B2 must freeze flow_action_query")
    if bool(freeze.get("unfreeze_lam_decoder", False)):
        raise RuntimeError("B1/B2 must keep LaWM frozen")
    if not bool(freeze.get("train_expert_latent_head", False)):
        raise RuntimeError("B1/B2 must train the expert latent head")

    # These shared modules must remain bitwise identical to the new Routing-V1 Base.
    shared_prefixes = (
        "policy_backend.act_query",
        "policy_backend.flow_action_query",
        "policy_backend.vlm_to_lam.",
        "policy_backend.lam.",
    )
    checked = 0
    for k, v in base.items():
        if not k.startswith(shared_prefixes):
            continue
        if k not in sd:
            raise RuntimeError(f"Shared frozen key missing from task checkpoint: {k}")
        assert_tensor_equal(v, sd[k], k)
        checked += 1
    if checked == 0:
        raise RuntimeError("No shared query/QFormer/LaWM tensors were checked")

    # B1 keeps the whole VLM bitwise frozen. B2 uses LoRA, so original base VLM
    # weights should still be bitwise unchanged in the *unmerged* checkpoint.
    vlm_checked = 0
    for k, v in base.items():
        if not k.startswith("policy_backend.vlm."):
            continue
        if k not in sd:
            continue
        assert_tensor_equal(v, sd[k], k)
        vlm_checked += 1
    if vlm_checked == 0:
        raise RuntimeError("No shared VLM tensors were checked")

    base_head = canonical_head_items(base)
    changed = []
    for k, v in base_head.items():
        if k in head and (v.shape != head[k].shape or not torch.equal(v, head[k])):
            changed.append(k)
    if not changed:
        raise RuntimeError("Expert latent head did not change relative to Routing-V1 Base")

    print(
        f"[OK] {args.mode.upper()} isolation: checked {checked} query/QFormer/LaWM tensors and "
        f"{vlm_checked} original VLM tensors; all frozen exactly."
    )
    print(f"[OK] task-specific expert latent head updated: changed_tensors={len(changed)}")


if __name__ == "__main__":
    main()
