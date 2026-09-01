from __future__ import annotations

from typing import Any, Dict

import torch


def map_policy_train_output(policy_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mapped = {
        "total_loss": policy_output["loss_total"],
        "loss_flow": policy_output["loss_flow"],
        "loss_perceptual": policy_output["loss_perceptual"],
        "loss_distill": policy_output["loss_distill"],
        "loss_vlm": policy_output["loss_vlm"],
    }
    # Routing-V1 auxiliary losses/diagnostics are optional so old checkpoints
    # and old experiment configs retain exactly the previous output contract.
    for key in (
        "loss_expert_latent",
        "loss_expert_world",
        "diag_zstar_to_zhat",
        "diag_zhat_to_zgt",
        "diag_hstar_to_hhat",
        "loss_routing_v2_semantic",
        "loss_routing_v2_dynamics",
        "loss_routing_v2_dynamics_gt",
        "loss_routing_v2_dynamics_pred",
        "diag_routing_v2_dyn_gt_delta",
        "diag_routing_v2_dyn_gt_z",
        "diag_routing_v2_dyn_pred_delta",
        "diag_routing_v2_dyn_pred_z",
    ):
        if key in policy_output:
            mapped[key] = policy_output[key]
    return mapped


def _tensor_to_numpy(value: torch.Tensor):
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


def _to_msgpackable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _tensor_to_numpy(value)
    if isinstance(value, dict):
        return {key: _to_msgpackable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_msgpackable(item) for item in value)
    return value


def map_policy_infer_output(
    actions: torch.Tensor,
    intermediates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {"normalized_actions": _tensor_to_numpy(actions)}
    if intermediates is not None:
        output["intermediates"] = _to_msgpackable(intermediates)
    return output
