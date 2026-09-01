#!/usr/bin/env python3
"""Preflight the first Small Flow S1 architecture without loading VLM/LAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from starVLA.model.framework.vlas.flowmatching_expert import (
    ConditionalFlowMatchingConfig,
    ConditionalFlowMatchingHead,
)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state_dict at {path}, got {type(state).__name__}.")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Optional Formal Base checkpoint used to report the Small/Full parameter ratio.",
    )
    args = parser.parse_args()

    config = ConditionalFlowMatchingConfig(
        expert_variant="small_s1",
        action_dim=32,
        state_dim=32,
        use_state=False,
        vlm_dim=2048,
        vision_dim=768,
        hidden_dim=512,
        num_layers=6,
        attention_heads=8,
        ffn_dim=2048,
        num_vision_tokens=256,
        num_target_vision_tokens=-1,
        interleave_self_attention=True,
        use_alternate_vldit=True,
        attend_text_every_n_blocks=2,
        cfg_drop_prob=0.0,
        cfg_guidance_scale=1.0,
        num_inference_steps=10,
        horizon_sec=0.4,
        use_action_positional_embeddings=True,
    )
    flow = ConditionalFlowMatchingHead(config=config)
    flow.action_horizon = 50

    total_params = sum(parameter.numel() for parameter in flow.parameters())
    trainable_params = sum(parameter.numel() for parameter in flow.parameters() if parameter.requires_grad)
    if total_params != trainable_params:
        raise RuntimeError(
            f"Fresh Small Flow should be fully trainable: total={total_params:,}, "
            f"trainable={trainable_params:,}."
        )
    if not 10_000_000 <= total_params <= 80_000_000:
        raise RuntimeError(
            f"Unexpected Small Flow size {total_params:,}; expected a first-stage 10M-80M expert."
        )

    module_params: list[tuple[str, int]] = []
    for name, module in flow.named_children():
        module_params.append((name, sum(parameter.numel() for parameter in module.parameters())))
    module_params.sort(key=lambda item: item[1], reverse=True)

    report: dict[str, object] = {
        "expert_variant": config.expert_variant,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "attention_heads": config.attention_heads,
        "ffn_dim": config.ffn_dim,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "largest_children": module_params,
    }

    if args.base_checkpoint is not None:
        state = _load_state_dict(args.base_checkpoint)
        base_flow_numel = sum(
            value.numel()
            for key, value in state.items()
            if key.startswith("policy_backend.flow.") and torch.is_tensor(value)
        )
        if base_flow_numel <= 0:
            raise RuntimeError(
                f"No canonical policy_backend.flow.* tensors found in {args.base_checkpoint}."
            )
        report["base_flow_numel"] = base_flow_numel
        report["small_to_base_ratio"] = total_params / base_flow_numel

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
