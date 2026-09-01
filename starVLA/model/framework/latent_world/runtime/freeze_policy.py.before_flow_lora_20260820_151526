from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from starVLA.model.framework.vlas.vlm_auto import (
    _keep_first_n_llm_layers,
    _resolve_llm_module,
    _unfreeze_last_n_llm_layers,
    freeze_qwen3vl,
)


@dataclass(frozen=True)
class LatentWorldPolicyFreezeConfig:
    # Original component-level freeze controls.
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_embedding: bool = False
    unfreeze_vision_merger: bool = False

    # Strict VLM-interface isolation controls used by prior diagnostics.
    freeze_vlm_all: bool = False
    freeze_act_query: bool = False
    freeze_flow_action_query: bool = False

    # NEW: hard diagnostic mode.
    #
    # When True, this is the FINAL AUTHORITY:
    #   - freeze every parameter under policy_backend
    #   - unfreeze ONLY policy_backend.flow.*
    #
    # This guarantees that VLM, merger, queries, QFormer/VLMToLAM,
    # LaWM encoder/decoder, and all other policy-backend parameters
    # stay fixed while a task-specific Flow action head is learned.
    train_flow_only: bool = False

    unfreeze_lam_decoder: bool = False
    keep_llm_first_n_layers: Optional[int] = None
    unfreeze_llm_last_n_layers: Optional[int] = None


def parse_policy_freeze_config(
    freeze_cfg: Any,
) -> LatentWorldPolicyFreezeConfig:
    if freeze_cfg is None:
        return LatentWorldPolicyFreezeConfig()

    unfreeze_last_n = freeze_cfg.get(
        "unfreeze_llm_last_n_layers",
        None,
    )
    if unfreeze_last_n is not None:
        unfreeze_last_n = int(unfreeze_last_n)

    keep_first_n = freeze_cfg.get(
        "keep_llm_first_n_layers",
        None,
    )
    if keep_first_n is not None:
        keep_first_n = int(keep_first_n)
        if keep_first_n <= 0:
            keep_first_n = None

    return LatentWorldPolicyFreezeConfig(
        freeze_vision_backbone=bool(
            freeze_cfg.get(
                "freeze_vision_backbone",
                False,
            )
        ),
        freeze_llm_backbone=bool(
            freeze_cfg.get(
                "freeze_llm_backbone",
                False,
            )
        ),
        freeze_last_llm_layer=bool(
            freeze_cfg.get(
                "freeze_last_llm_layer",
                False,
            )
        ),
        freeze_embedding=bool(
            freeze_cfg.get(
                "freeze_embedding",
                False,
            )
        ),
        unfreeze_vision_merger=bool(
            freeze_cfg.get(
                "unfreeze_vision_merger",
                False,
            )
        ),

        freeze_vlm_all=bool(
            freeze_cfg.get(
                "freeze_vlm_all",
                False,
            )
        ),
        freeze_act_query=bool(
            freeze_cfg.get(
                "freeze_act_query",
                False,
            )
        ),
        freeze_flow_action_query=bool(
            freeze_cfg.get(
                "freeze_flow_action_query",
                False,
            )
        ),

        train_flow_only=bool(
            freeze_cfg.get(
                "train_flow_only",
                False,
            )
        ),

        unfreeze_lam_decoder=bool(
            freeze_cfg.get(
                "unfreeze_lam_decoder",
                False,
            )
        ),
        keep_llm_first_n_layers=keep_first_n,
        unfreeze_llm_last_n_layers=unfreeze_last_n,
    )


def _apply_strict_vlm_interface_freeze(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """Apply the strict VLM/interface diagnostic freeze."""

    if freeze_policy.freeze_vlm_all:
        policy_backend.vlm.requires_grad_(False)

    if freeze_policy.freeze_act_query:
        act_query = getattr(
            policy_backend,
            "act_query",
            None,
        )
        if act_query is None:
            raise RuntimeError(
                "freeze_act_query=True but "
                "policy_backend.act_query does not exist."
            )
        act_query.requires_grad_(False)

    if freeze_policy.freeze_flow_action_query:
        flow_action_query = getattr(
            policy_backend,
            "flow_action_query",
            None,
        )
        if flow_action_query is None:
            raise RuntimeError(
                "freeze_flow_action_query=True but "
                "policy_backend.flow_action_query "
                "does not exist."
            )
        flow_action_query.requires_grad_(False)

    # Hard runtime audit.
    if freeze_policy.freeze_vlm_all:
        leaking = [
            name
            for name, p
            in policy_backend.vlm.named_parameters()
            if p.requires_grad
        ]

        if leaking:
            raise RuntimeError(
                "freeze_vlm_all=True but VLM still "
                "has trainable parameters: "
                f"{leaking[:20]}"
            )

    if freeze_policy.freeze_act_query:
        if getattr(
            policy_backend,
            "act_query",
        ).requires_grad:
            raise RuntimeError(
                "act_query is unexpectedly still trainable."
            )

    if freeze_policy.freeze_flow_action_query:
        if getattr(
            policy_backend,
            "flow_action_query",
        ).requires_grad:
            raise RuntimeError(
                "flow_action_query is unexpectedly "
                "still trainable."
            )


def _apply_flow_only_training(
    policy_backend,
) -> None:
    """
    Freeze the ENTIRE policy backend and reopen ONLY Flow.

    This is deliberately implemented as a final override instead of
    trying to enumerate every upstream module one by one.  Therefore
    newly added upstream modules will also be frozen automatically.
    """

    # Freeze absolutely everything in the policy backend first.
    policy_backend.requires_grad_(False)

    flow = getattr(
        policy_backend,
        "flow",
        None,
    )

    if flow is None:
        raise RuntimeError(
            "train_flow_only=True but "
            "policy_backend.flow does not exist."
        )

    # Re-open ONLY the action Flow head.
    flow.requires_grad_(True)

    # Hard parameter-level audit.
    trainable = [
        name
        for name, p
        in policy_backend.named_parameters()
        if p.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "train_flow_only=True but no trainable "
            "Flow parameters were found."
        )

    unexpected = [
        name
        for name in trainable
        if not (
            name == "flow"
            or name.startswith("flow.")
        )
    ]

    if unexpected:
        raise RuntimeError(
            "train_flow_only=True but non-Flow "
            "parameters are still trainable: "
            f"{unexpected[:50]}"
        )

    flow_trainable = [
        name
        for name in trainable
        if (
            name == "flow"
            or name.startswith("flow.")
        )
    ]

    print(
        "[freeze-policy] FLOW-ONLY training enabled: "
        f"trainable_tensors={len(flow_trainable)}, "
        "all non-Flow policy parameters frozen."
    )


def apply_policy_freeze(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    # ------------------------------------------------------------------
    # Original VLM freeze logic.
    # ------------------------------------------------------------------
    freeze_qwen3vl(
        policy_backend.vlm,
        freeze_vision_backbone=(
            freeze_policy.freeze_vision_backbone
        ),
        freeze_llm_backbone=(
            freeze_policy.freeze_llm_backbone
        ),
        freeze_last_llm_layer=(
            freeze_policy.freeze_last_llm_layer
        ),
        freeze_embedding=(
            freeze_policy.freeze_embedding
        ),
        unfreeze_vision_merger=(
            freeze_policy.unfreeze_vision_merger
        ),
    )

    llm_module = None

    if (
        freeze_policy.keep_llm_first_n_layers
        is not None
        or (
            freeze_policy.freeze_llm_backbone
            and (
                freeze_policy.unfreeze_llm_last_n_layers
                is not None
            )
            and (
                freeze_policy.unfreeze_llm_last_n_layers
                > 0
            )
        )
    ):
        llm_module = _resolve_llm_module(
            policy_backend.vlm
        )

    if (
        freeze_policy.keep_llm_first_n_layers
        is not None
    ):
        _keep_first_n_llm_layers(
            llm_module,
            freeze_policy.keep_llm_first_n_layers,
        )

    if (
        freeze_policy.freeze_llm_backbone
        and (
            freeze_policy.unfreeze_llm_last_n_layers
            is not None
        )
        and (
            freeze_policy.unfreeze_llm_last_n_layers
            > 0
        )
    ):
        _unfreeze_last_n_llm_layers(
            llm_module,
            freeze_policy.unfreeze_llm_last_n_layers,
        )

    # ------------------------------------------------------------------
    # Prior strict VLM-interface isolation logic.
    # ------------------------------------------------------------------
    _apply_strict_vlm_interface_freeze(
        policy_backend,
        freeze_policy,
    )

    # ------------------------------------------------------------------
    # Original LAM freeze logic.
    # ------------------------------------------------------------------
    for p in policy_backend.lam.parameters():
        p.requires_grad = False

    if freeze_policy.unfreeze_lam_decoder:
        lam_decoder = getattr(
            policy_backend.lam,
            "decoder",
            None,
        )
        if lam_decoder is not None:
            for p in lam_decoder.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # NEW FINAL AUTHORITY: Flow-only training.
    #
    # This MUST stay at the end.  It overrides every earlier component-
    # level freeze/unfreeze decision and guarantees that only Flow can
    # receive optimizer updates.
    # ------------------------------------------------------------------
    if freeze_policy.train_flow_only:
        _apply_flow_only_training(
            policy_backend,
        )