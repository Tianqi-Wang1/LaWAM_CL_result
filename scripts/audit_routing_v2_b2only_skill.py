#!/usr/bin/env python3
"""Strictly audit a simplified Routing-V2 B2-only skill checkpoint.

Expected task-specific parameters:
  * VLM text LoRA, rank 32
  * Flow DiT blocks 12-15, dense
  * Flow conditioning adapters, bottleneck 128
  * Flow nonlinear adapters in blocks 0-11, bottleneck 128

Query residuals, QFormer LoRA, LaWM LoRA, routing memories, and changes to
every other Base tensor are rejected.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import torch


TEXT_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
DENSE_LAYERS = set(range(12, 16))
NONLINEAR_LAYERS = set(range(0, 12))
EXPECTED = {
    "dense": 57_716_736,
    "vlm_text_lora": 19_922_944,
    "conditioning": 6_651_904,
    "nonlinear": 6_291_456,
    "total": 90_583_040,
}


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
        for key in ("state_dict", "model", "module"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and all(torch.is_tensor(v) for v in inner.values()):
                return dict(inner)
    raise RuntimeError(f"Unsupported checkpoint structure: {path}")


def canonical_key(key: str) -> str:
    if key.startswith("policy_action_head."):
        return "policy_backend.flow." + key[len("policy_action_head."):]
    if key.startswith("policy_vlm_adapter.model."):
        return "policy_backend.vlm." + key[len("policy_vlm_adapter.model."):]
    return key


def canonical_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = canonical_key(key)
        if key in out and not torch.equal(out[key], value):
            raise RuntimeError(f"Alias mismatch for {key}")
        out[key] = value
    return out


def flow_block_idx(key: str):
    match = re.match(r"^policy_backend\.flow\.DiT\.transformer_blocks\.(\d+)\.", key)
    return None if match is None else int(match.group(1))


def is_text_lora(key: str) -> bool:
    if not (key.endswith(".lora_A") or key.endswith(".lora_B")):
        return False
    prefix = key.rsplit(".lora_", 1)[0]
    return (
        prefix.startswith("policy_backend.vlm.")
        and ".language_model.layers." in prefix
        and any(prefix.endswith(suffix) for suffix in TEXT_SUFFIXES)
    )


def is_conditioning(key: str) -> bool:
    return key.startswith("policy_backend.flow.") and ".conditioning_adapter." in key


def is_nonlinear(key: str) -> bool:
    return key.startswith("policy_backend.flow.") and (
        ".attn_nonlinear_adapter." in key or ".ffn_nonlinear_adapter." in key
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base = canonical_state(load_state(args.base))
    skill = canonical_state(load_state(args.skill))
    missing = [key for key in base if key not in skill]
    if missing:
        raise RuntimeError(f"Skill checkpoint is missing Base tensors: {missing[:20]}")

    dense_budget = sum(
        value.numel()
        for key, value in base.items()
        if flow_block_idx(key) in DENSE_LAYERS
    )
    changed_dense = []
    unexpected_base_changes = []
    for key, base_value in base.items():
        if torch.equal(base_value, skill[key]):
            continue
        if flow_block_idx(key) in DENSE_LAYERS:
            changed_dense.append(key)
        else:
            unexpected_base_changes.append(key)
    if unexpected_base_changes:
        raise RuntimeError(
            "B2-only isolation failed; non-Last4 Base tensors changed: "
            f"{unexpected_base_changes[:20]}"
        )
    if not changed_dense:
        raise RuntimeError("No dense tensor in Flow blocks 12-15 changed")

    extras = {key: value for key, value in skill.items() if key not in base}
    text_lora = {key: value for key, value in extras.items() if is_text_lora(key)}
    conditioning = {key: value for key, value in extras.items() if is_conditioning(key)}
    nonlinear = {key: value for key, value in extras.items() if is_nonlinear(key)}
    allowed = set(text_lora) | set(conditioning) | set(nonlinear)
    unexpected_extras = sorted(set(extras) - allowed)
    if unexpected_extras:
        raise RuntimeError(
            "B2-only checkpoint contains unexpected task structures "
            "(for example Query/QFormer/WM adapters): "
            f"{unexpected_extras[:20]}"
        )

    for key, value in text_lora.items():
        if key.endswith(".lora_A") and int(value.shape[0]) != 32:
            raise RuntimeError(f"Wrong Text-LoRA rank for {key}: {tuple(value.shape)}")
    if not any(
        key.endswith(".lora_B") and torch.count_nonzero(value).item() > 0
        for key, value in text_lora.items()
    ):
        raise RuntimeError("All Text-LoRA B matrices are still zero")

    nonlinear_layers = {flow_block_idx(key) for key in nonlinear}
    nonlinear_layers.discard(None)
    if nonlinear_layers != NONLINEAR_LAYERS:
        raise RuntimeError(
            f"Nonlinear adapter layers={sorted(nonlinear_layers)}, "
            f"expected={sorted(NONLINEAR_LAYERS)}"
        )
    for layer in NONLINEAR_LAYERS:
        keys = [key for key in nonlinear if flow_block_idx(key) == layer]
        if not any(".attn_nonlinear_adapter." in key for key in keys):
            raise RuntimeError(f"Layer {layer} has no attention nonlinear adapter")
        if not any(".ffn_nonlinear_adapter." in key for key in keys):
            raise RuntimeError(f"Layer {layer} has no FFN nonlinear adapter")

    for label, state in (("conditioning", conditioning), ("nonlinear", nonlinear)):
        ups = [value for key, value in state.items() if key.endswith(".up.weight")]
        if not ups or not any(torch.count_nonzero(value).item() > 0 for value in ups):
            raise RuntimeError(f"All {label} adapter up projections are still zero")

    counts = {
        "dense": int(dense_budget),
        "vlm_text_lora": int(sum(value.numel() for value in text_lora.values())),
        "conditioning": int(sum(value.numel() for value in conditioning.values())),
        "nonlinear": int(sum(value.numel() for value in nonlinear.values())),
    }
    counts["total"] = sum(counts.values())
    if counts != EXPECTED:
        raise RuntimeError(f"B2-only parameter budget mismatch: got={counts}, expected={EXPECTED}")

    if args.config is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        action_cfg = cfg.framework.action_model
        freeze_cfg = cfg.trainer.freeze
        checks = {
            "routing_v2_enable_query_delta": bool(action_cfg.routing_v2_enable_query_delta),
            "train_routing_v2_skill": bool(freeze_cfg.train_routing_v2_skill),
            "routing_v2_b2_only_skill_path": bool(freeze_cfg.routing_v2_b2_only_skill_path),
            "train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter": bool(
                freeze_cfg.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter
            ),
            "unfreeze_lam_decoder": bool(freeze_cfg.unfreeze_lam_decoder),
        }
        expected_checks = {
            "routing_v2_enable_query_delta": False,
            "train_routing_v2_skill": False,
            "routing_v2_b2_only_skill_path": True,
            "train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter": True,
            "unfreeze_lam_decoder": False,
        }
        if checks != expected_checks:
            raise RuntimeError(f"B2-only config mismatch: got={checks}, expected={expected_checks}")

    report = {
        "base": str(args.base.resolve()),
        "skill": str(args.skill.resolve()),
        "parameter_groups": counts,
        "changed_dense_tensor_count": len(changed_dense),
        "nonlinear_layers": sorted(nonlinear_layers),
        "forbidden_task_structures": {
            "query_delta": 0,
            "qformer_lora": 0,
            "lawm_lora": 0,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[OK] Simplified Routing-V2 B2-only skill audit passed")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
