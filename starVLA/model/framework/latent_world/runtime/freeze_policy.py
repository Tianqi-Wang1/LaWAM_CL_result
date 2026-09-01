from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from starVLA.model.framework.vlas.vlm_auto import (
    _keep_first_n_llm_layers,
    _resolve_llm_module,
    _unfreeze_last_n_llm_layers,
    freeze_qwen3vl,
)
from starVLA.model.framework.latent_world.runtime.flow_lora import (
    inject_flow_conditioning_lora,
    inject_flow_interface_lora,
)
from starVLA.model.framework.latent_world.runtime.vlm_lora import inject_vlm_lora
from starVLA.model.framework.latent_world.routing_v2 import (
    inject_recursive_linear_lora,
    set_lora_trainable,
)


@dataclass(frozen=True)
class LatentWorldPolicyFreezeConfig:
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_embedding: bool = False
    unfreeze_vision_merger: bool = False

    freeze_vlm_all: bool = False
    freeze_act_query: bool = False
    freeze_flow_action_query: bool = False

    train_flow_only: bool = False
    train_flow_lora: bool = False
    train_flow_residual_expert: bool = False
    # Dense fine-tuning of selected ORIGINAL DiT transformer blocks only.
    # No new expert modules are created in this mode.
    train_flow_partial_dense: bool = False
    flow_partial_dense_layer_indices: Optional[tuple[int, ...]] = None

    # Optional Flow I/O-interface adaptation paired with partial-dense DiT FT.
    # Dense and LoRA interface modes target the SAME functional modules.
    train_flow_interface_dense: bool = False
    train_flow_interface_lora: bool = False
    flow_interface_target_enc_vlm: bool = True
    flow_interface_target_action_encoder: bool = True
    flow_interface_target_action_decoder: bool = True
    flow_interface_target_state_encoder: bool = True
    flow_interface_target_output: bool = True
    flow_interface_target_timestep: bool = True

    # Joint VLM + Action-Flow LoRA diagnostic.  The action side is fixed to
    # all 16 original DiT blocks (Q/K/V/O + FFN only); VLM target location is
    # the experimental variable.
    train_vlm_flow_lora: bool = False
    # Hybrid diagnostics used in the E1/E7 single-task study.
    # E1: VLM text LoRA + selected ORIGINAL Flow DiT blocks dense FT.
    train_vlm_text_lora_partial_dense: bool = False
    # E7: VLM text LoRA + CLARE-style conditioning adapters in enc_vlm and AdaLN.
    train_vlm_text_lora_conditioning_adapter: bool = False
    # Compact follow-up modes.
    # Action-only: selected original DiT blocks dense FT + conditioning adapters.
    train_flow_partial_dense_conditioning_adapter: bool = False
    # Hybrid: selected original DiT blocks dense FT + conditioning adapters + VLM text LoRA.
    train_vlm_text_lora_partial_dense_conditioning_adapter: bool = False

    # v8 activation-space nonlinear adapter modes.
    # Action-only: selected original DiT blocks dense FT + conditioning adapters +
    # nonlinear Attention/FFN side adapters in the configured DiT layers.
    train_flow_partial_dense_conditioning_nonlinear_adapter: bool = False
    # Same as above, plus VLM text LoRA.
    train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter: bool = False
    # No dense DiT FT: conditioning + nonlinear DiT side adapters only.
    train_flow_conditioning_nonlinear_adapter: bool = False

    # Routing-V1 auxiliary latent head.  This flag only controls trainability;
    # the structural head itself is created by flow_cfg.enable_expert_latent_head.
    train_expert_latent_head: bool = False

    # Routing-V2 modes. Skill mode = B2 action expert + VLM text LoRA +
    # task-query residuals + QFormer Linear-LoRA + LaWM Linear-LoRA.
    # Memory mode freezes the entire skill and trains only the two AEs.
    train_routing_v2_skill: bool = False
    train_routing_v2_memory_only: bool = False
    routing_v2_qformer_lora_rank: int = 32
    routing_v2_qformer_lora_alpha: float = 32.0
    routing_v2_qformer_lora_dropout: float = 0.0
    routing_v2_lawm_lora_rank: int = 32
    routing_v2_lawm_lora_alpha: float = 32.0
    routing_v2_lawm_lora_dropout: float = 0.0

    vlm_lora_target_text: bool = False
    vlm_lora_text_last_n: int = 0
    vlm_lora_target_vision: bool = False
    vlm_lora_target_merger: bool = False
    vlm_lora_rank: int = 8
    vlm_lora_alpha: float = 8.0
    vlm_lora_dropout: float = 0.0

    flow_lora_rank: int = 8
    flow_lora_alpha: float = 8.0
    flow_lora_dropout: float = 0.0

    # Kept for compatibility; this package always uses the Level-2 core.
    flow_lora_target_attention: bool = True
    flow_lora_target_ffn: bool = True
    flow_lora_target_enc_vlm: bool = True
    flow_lora_target_output: bool = True

    # New conditioning-path controls.
    flow_lora_target_adanorm: bool = False
    flow_lora_target_timestep: bool = False

    unfreeze_lam_decoder: bool = False
    keep_llm_first_n_layers: Optional[int] = None
    unfreeze_llm_last_n_layers: Optional[int] = None


def parse_policy_freeze_config(
    freeze_cfg: Any,
) -> LatentWorldPolicyFreezeConfig:
    if freeze_cfg is None:
        return LatentWorldPolicyFreezeConfig()

    unfreeze_last_n = freeze_cfg.get(
        "unfreeze_llm_last_n_layers", None
    )
    if unfreeze_last_n is not None:
        unfreeze_last_n = int(unfreeze_last_n)

    keep_first_n = freeze_cfg.get(
        "keep_llm_first_n_layers", None
    )
    if keep_first_n is not None:
        keep_first_n = int(keep_first_n)
        if keep_first_n <= 0:
            keep_first_n = None

    raw_partial_layers = freeze_cfg.get(
        "flow_partial_dense_layer_indices", None
    )
    partial_layers = None
    if raw_partial_layers is not None:
        if isinstance(raw_partial_layers, str):
            text = raw_partial_layers.strip().strip("[]")
            values = [] if not text else [x.strip() for x in text.split(",")]
        else:
            values = list(raw_partial_layers)
        partial_layers = tuple(int(x) for x in values)
        if len(partial_layers) == 0:
            partial_layers = None

    return LatentWorldPolicyFreezeConfig(
        freeze_vision_backbone=bool(
            freeze_cfg.get("freeze_vision_backbone", False)
        ),
        freeze_llm_backbone=bool(
            freeze_cfg.get("freeze_llm_backbone", False)
        ),
        freeze_last_llm_layer=bool(
            freeze_cfg.get("freeze_last_llm_layer", False)
        ),
        freeze_embedding=bool(
            freeze_cfg.get("freeze_embedding", False)
        ),
        unfreeze_vision_merger=bool(
            freeze_cfg.get("unfreeze_vision_merger", False)
        ),
        freeze_vlm_all=bool(
            freeze_cfg.get("freeze_vlm_all", False)
        ),
        freeze_act_query=bool(
            freeze_cfg.get("freeze_act_query", False)
        ),
        freeze_flow_action_query=bool(
            freeze_cfg.get("freeze_flow_action_query", False)
        ),
        train_flow_only=bool(
            freeze_cfg.get("train_flow_only", False)
        ),
        train_flow_lora=bool(
            freeze_cfg.get("train_flow_lora", False)
        ),
        train_flow_residual_expert=bool(
            freeze_cfg.get("train_flow_residual_expert", False)
        ),
        train_flow_partial_dense=bool(
            freeze_cfg.get("train_flow_partial_dense", False)
        ),
        flow_partial_dense_layer_indices=partial_layers,
        train_flow_interface_dense=bool(
            freeze_cfg.get("train_flow_interface_dense", False)
        ),
        train_flow_interface_lora=bool(
            freeze_cfg.get("train_flow_interface_lora", False)
        ),
        flow_interface_target_enc_vlm=bool(
            freeze_cfg.get("flow_interface_target_enc_vlm", True)
        ),
        flow_interface_target_action_encoder=bool(
            freeze_cfg.get("flow_interface_target_action_encoder", True)
        ),
        flow_interface_target_action_decoder=bool(
            freeze_cfg.get("flow_interface_target_action_decoder", True)
        ),
        flow_interface_target_state_encoder=bool(
            freeze_cfg.get("flow_interface_target_state_encoder", True)
        ),
        flow_interface_target_output=bool(
            freeze_cfg.get("flow_interface_target_output", True)
        ),
        flow_interface_target_timestep=bool(
            freeze_cfg.get("flow_interface_target_timestep", True)
        ),
        train_vlm_flow_lora=bool(
            freeze_cfg.get("train_vlm_flow_lora", False)
        ),
        train_vlm_text_lora_partial_dense=bool(
            freeze_cfg.get("train_vlm_text_lora_partial_dense", False)
        ),
        train_vlm_text_lora_conditioning_adapter=bool(
            freeze_cfg.get("train_vlm_text_lora_conditioning_adapter", False)
        ),
        train_flow_partial_dense_conditioning_adapter=bool(
            freeze_cfg.get("train_flow_partial_dense_conditioning_adapter", False)
        ),
        train_vlm_text_lora_partial_dense_conditioning_adapter=bool(
            freeze_cfg.get("train_vlm_text_lora_partial_dense_conditioning_adapter", False)
        ),
        train_flow_partial_dense_conditioning_nonlinear_adapter=bool(
            freeze_cfg.get("train_flow_partial_dense_conditioning_nonlinear_adapter", False)
        ),
        train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter=bool(
            freeze_cfg.get("train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter", False)
        ),
        train_flow_conditioning_nonlinear_adapter=bool(
            freeze_cfg.get("train_flow_conditioning_nonlinear_adapter", False)
        ),
        train_expert_latent_head=bool(
            freeze_cfg.get("train_expert_latent_head", False)
        ),
        train_routing_v2_skill=bool(
            freeze_cfg.get("train_routing_v2_skill", False)
        ),
        train_routing_v2_memory_only=bool(
            freeze_cfg.get("train_routing_v2_memory_only", False)
        ),
        routing_v2_qformer_lora_rank=int(
            freeze_cfg.get("routing_v2_qformer_lora_rank", 32)
        ),
        routing_v2_qformer_lora_alpha=float(
            freeze_cfg.get("routing_v2_qformer_lora_alpha", 32.0)
        ),
        routing_v2_qformer_lora_dropout=float(
            freeze_cfg.get("routing_v2_qformer_lora_dropout", 0.0)
        ),
        routing_v2_lawm_lora_rank=int(
            freeze_cfg.get("routing_v2_lawm_lora_rank", 32)
        ),
        routing_v2_lawm_lora_alpha=float(
            freeze_cfg.get("routing_v2_lawm_lora_alpha", 32.0)
        ),
        routing_v2_lawm_lora_dropout=float(
            freeze_cfg.get("routing_v2_lawm_lora_dropout", 0.0)
        ),
        vlm_lora_target_text=bool(
            freeze_cfg.get("vlm_lora_target_text", False)
        ),
        vlm_lora_text_last_n=int(
            freeze_cfg.get("vlm_lora_text_last_n", 0)
        ),
        vlm_lora_target_vision=bool(
            freeze_cfg.get("vlm_lora_target_vision", False)
        ),
        vlm_lora_target_merger=bool(
            freeze_cfg.get("vlm_lora_target_merger", False)
        ),
        vlm_lora_rank=int(
            freeze_cfg.get("vlm_lora_rank", 8)
        ),
        vlm_lora_alpha=float(
            freeze_cfg.get("vlm_lora_alpha", 8.0)
        ),
        vlm_lora_dropout=float(
            freeze_cfg.get("vlm_lora_dropout", 0.0)
        ),
        flow_lora_rank=int(
            freeze_cfg.get("flow_lora_rank", 8)
        ),
        flow_lora_alpha=float(
            freeze_cfg.get("flow_lora_alpha", 8.0)
        ),
        flow_lora_dropout=float(
            freeze_cfg.get("flow_lora_dropout", 0.0)
        ),
        flow_lora_target_attention=bool(
            freeze_cfg.get("flow_lora_target_attention", True)
        ),
        flow_lora_target_ffn=bool(
            freeze_cfg.get("flow_lora_target_ffn", True)
        ),
        flow_lora_target_enc_vlm=bool(
            freeze_cfg.get("flow_lora_target_enc_vlm", True)
        ),
        flow_lora_target_output=bool(
            freeze_cfg.get("flow_lora_target_output", True)
        ),
        flow_lora_target_adanorm=bool(
            freeze_cfg.get("flow_lora_target_adanorm", False)
        ),
        flow_lora_target_timestep=bool(
            freeze_cfg.get("flow_lora_target_timestep", False)
        ),
        unfreeze_lam_decoder=bool(
            freeze_cfg.get("unfreeze_lam_decoder", False)
        ),
        keep_llm_first_n_layers=keep_first_n,
        unfreeze_llm_last_n_layers=unfreeze_last_n,
    )


def _apply_strict_vlm_interface_freeze(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    if freeze_policy.freeze_vlm_all:
        policy_backend.vlm.requires_grad_(False)

    if freeze_policy.freeze_act_query:
        obj = getattr(policy_backend, "act_query", None)
        if obj is None:
            raise RuntimeError(
                "freeze_act_query=True but act_query does not exist."
            )
        obj.requires_grad_(False)

    if freeze_policy.freeze_flow_action_query:
        obj = getattr(
            policy_backend,
            "flow_action_query",
            None,
        )
        if obj is None:
            raise RuntimeError(
                "freeze_flow_action_query=True but flow_action_query "
                "does not exist."
            )
        obj.requires_grad_(False)


def _apply_flow_only_training(policy_backend) -> None:
    policy_backend.requires_grad_(False)

    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError(
            "train_flow_only=True but policy_backend.flow does not exist."
        )

    flow.requires_grad_(True)

    trainable = [
        name
        for name, p in policy_backend.named_parameters()
        if p.requires_grad
    ]
    if not trainable:
        raise RuntimeError(
            "train_flow_only=True but no Flow parameters are trainable."
        )

    bad = [
        name
        for name in trainable
        if not (name == "flow" or name.startswith("flow."))
    ]
    if bad:
        raise RuntimeError(
            f"Non-Flow parameters remain trainable: {bad[:50]}"
        )


def _apply_flow_conditioning_lora(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    if freeze_policy.train_flow_only:
        raise RuntimeError(
            "train_flow_only and train_flow_lora are mutually exclusive."
        )

    # Final authority: freeze every pre-existing Base parameter.
    policy_backend.requires_grad_(False)

    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError(
            "train_flow_lora=True but policy_backend.flow does not exist."
        )

    # This experiment always includes the fresh Level-2 reference core.
    if not all(
        (
            freeze_policy.flow_lora_target_attention,
            freeze_policy.flow_lora_target_ffn,
            freeze_policy.flow_lora_target_enc_vlm,
            freeze_policy.flow_lora_target_output,
        )
    ):
        raise RuntimeError(
            "Conditioning experiment requires the complete Level-2 core: "
            "attention + FFN + enc_vlm + output."
        )

    summary = inject_flow_conditioning_lora(
        flow,
        rank=freeze_policy.flow_lora_rank,
        alpha=freeze_policy.flow_lora_alpha,
        dropout=freeze_policy.flow_lora_dropout,
        target_attention=freeze_policy.flow_lora_target_attention,
        target_ffn=freeze_policy.flow_lora_target_ffn,
        target_enc_vlm=freeze_policy.flow_lora_target_enc_vlm,
        target_output=freeze_policy.flow_lora_target_output,
        target_adanorm=freeze_policy.flow_lora_target_adanorm,
        target_timestep=freeze_policy.flow_lora_target_timestep,
    )

    # ONLY newly introduced LoRA tensors are trainable.
    for name, p in flow.named_parameters():
        p.requires_grad = bool(
            name.endswith(".lora_A")
            or name.endswith(".lora_B")
        )

    trainable = [
        (name, p)
        for name, p in policy_backend.named_parameters()
        if p.requires_grad
    ]

    unexpected = [
        name
        for name, _ in trainable
        if not (
            name.startswith("flow.")
            and (
                name.endswith(".lora_A")
                or name.endswith(".lora_B")
            )
        )
    ]
    if unexpected:
        raise RuntimeError(
            "FLOW-LoRA isolation failed; non-LoRA params trainable: "
            f"{unexpected[:50]}"
        )

    trainable_params = sum(p.numel() for _, p in trainable)
    if trainable_params != summary.trainable_params:
        raise RuntimeError(
            f"Trainable parameter mismatch: "
            f"runtime={trainable_params}, "
            f"summary={summary.trainable_params}"
        )

    print(
        "[freeze-policy] FLOW CONDITIONING-LORA enabled: "
        f"target_modules={len(summary.target_modules)}, "
        f"trainable_tensors={summary.trainable_tensors}, "
        f"trainable_params={summary.trainable_params:,}, "
        f"rank={summary.rank}, alpha={summary.alpha}, "
        f"adanorm={summary.target_adanorm}, "
        f"timestep={summary.target_timestep}; "
        f"groups=(attention={summary.attention_targets}, "
        f"ffn={summary.ffn_targets}, "
        f"enc_vlm={summary.enc_vlm_targets}, "
        f"output={summary.output_targets}, "
        f"adanorm={summary.adanorm_targets}, "
        f"timestep={summary.timestep_targets}); "
        "EVERY Base WAM/Flow parameter frozen."
    )



def _apply_flow_residual_expert_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """Freeze the full Base/WAM/Flow model and train only new residual expert blocks."""
    policy_backend.requires_grad_(False)

    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError(
            "train_flow_residual_expert=True but policy_backend.flow does not exist."
        )

    dit = getattr(flow, "DiT", None)
    residual_blocks = getattr(dit, "residual_expert_blocks", None) if dit is not None else None
    num_blocks = int(getattr(dit, "residual_expert_num_blocks", 0)) if dit is not None else 0
    if residual_blocks is None or num_blocks <= 0 or len(residual_blocks) != num_blocks:
        raise RuntimeError(
            "train_flow_residual_expert=True requires "
            "framework.action_model.flow_cfg.residual_expert_num_blocks > 0."
        )

    residual_blocks.requires_grad_(True)

    trainable = [
        (name, p)
        for name, p in policy_backend.named_parameters()
        if p.requires_grad
    ]
    if not trainable:
        raise RuntimeError(
            "train_flow_residual_expert=True but no residual expert parameters are trainable."
        )

    prefix = "flow.DiT.residual_expert_blocks."
    unexpected = [name for name, _ in trainable if not name.startswith(prefix)]
    if unexpected:
        raise RuntimeError(
            "Residual-Expert isolation failed; non-residual params trainable: "
            f"{unexpected[:50]}"
        )

    trainable_params = sum(p.numel() for _, p in trainable)
    print(
        "[FLOW-Residual] isolation active: "
        f"blocks={num_blocks}, trainable_tensors={len(trainable)}, "
        f"trainable_params={trainable_params:,}; "
        "ALL pre-existing Base/WAM/Flow parameters frozen."
    )


def _apply_flow_residual_plus_lora_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """Train residual experts plus LoRA on selected ORIGINAL Flow modules only."""
    if freeze_policy.train_flow_only:
        raise RuntimeError(
            "train_flow_only cannot be combined with residual+LoRA training."
        )

    policy_backend.requires_grad_(False)
    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError(
            "Residual+LoRA mode requires policy_backend.flow."
        )

    dit = getattr(flow, "DiT", None)
    residual_blocks = getattr(dit, "residual_expert_blocks", None) if dit is not None else None
    num_blocks = int(getattr(dit, "residual_expert_num_blocks", 0)) if dit is not None else 0
    if residual_blocks is None or num_blocks <= 0 or len(residual_blocks) != num_blocks:
        raise RuntimeError(
            "Residual+LoRA mode requires at least one residual expert block."
        )

    summary = inject_flow_conditioning_lora(
        flow,
        rank=freeze_policy.flow_lora_rank,
        alpha=freeze_policy.flow_lora_alpha,
        dropout=freeze_policy.flow_lora_dropout,
        target_attention=freeze_policy.flow_lora_target_attention,
        target_ffn=freeze_policy.flow_lora_target_ffn,
        target_enc_vlm=freeze_policy.flow_lora_target_enc_vlm,
        target_output=freeze_policy.flow_lora_target_output,
        target_adanorm=freeze_policy.flow_lora_target_adanorm,
        target_timestep=freeze_policy.flow_lora_target_timestep,
    )

    # Reassert strict isolation after module replacement.
    for name, param in flow.named_parameters():
        is_lora = name.endswith(".lora_A") or name.endswith(".lora_B")
        is_residual = name.startswith("DiT.residual_expert_blocks.")
        param.requires_grad = bool(is_lora or is_residual)

    trainable = [
        (name, p)
        for name, p in policy_backend.named_parameters()
        if p.requires_grad
    ]
    unexpected = [
        name for name, _ in trainable
        if not (
            name.startswith("flow.DiT.residual_expert_blocks.")
            or (
                name.startswith("flow.")
                and (name.endswith(".lora_A") or name.endswith(".lora_B"))
            )
        )
    ]
    if unexpected:
        raise RuntimeError(
            "Residual+LoRA isolation failed; unexpected trainable params: "
            f"{unexpected[:50]}"
        )

    residual_params = sum(
        p.numel() for name, p in trainable
        if name.startswith("flow.DiT.residual_expert_blocks.")
    )
    lora_params = sum(
        p.numel() for name, p in trainable
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    )
    if lora_params != summary.trainable_params:
        raise RuntimeError(
            "Residual+LoRA LoRA parameter mismatch: "
            f"runtime={lora_params}, summary={summary.trainable_params}"
        )

    layer_indices = tuple(getattr(dit, "residual_expert_layer_indices", ()))
    print(
        "[FLOW-Residual+LoRA] isolation active: "
        f"residual_layers={layer_indices}, residual_params={residual_params:,}, "
        f"lora_targets={len(summary.target_modules)}, lora_params={lora_params:,}, "
        f"rank={summary.rank}, alpha={summary.alpha}; "
        f"groups=(attention={summary.attention_targets}, "
        f"ffn={summary.ffn_targets}, enc_vlm={summary.enc_vlm_targets}, "
        f"output={summary.output_targets}, adanorm={summary.adanorm_targets}, "
        f"timestep={summary.timestep_targets}); "
        "ALL original Base/WAM/Flow weights frozen."
    )


def _flow_interface_dense_modules(flow):
    modules = []
    for name in ("enc_vlm", "action_encoder", "action_decoder"):
        obj = getattr(flow, name, None)
        if obj is None:
            raise RuntimeError(f"Flow interface module missing: flow.{name}")
        modules.append((name, obj))

    enc_state = getattr(flow, "enc_state", None)
    if enc_state is not None:
        modules.append(("enc_state", enc_state))

    dit = getattr(flow, "DiT", None)
    if dit is None:
        raise RuntimeError("Flow interface requires flow.DiT")
    for name in ("proj_out_1", "proj_out_2", "timestep_encoder"):
        obj = getattr(dit, name, None)
        if obj is None:
            raise RuntimeError(f"Flow interface module missing: flow.DiT.{name}")
        modules.append((f"DiT.{name}", obj))
    return modules


def _apply_flow_partial_dense_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """Full-rank FT selected ORIGINAL DiT blocks, optionally adapting Flow I/O interface.

    Interface-Dense and Interface-LoRA modes target the same functional path:
      enc_vlm, action_encoder, action_decoder, optional enc_state,
      DiT.proj_out_1/2, and DiT.timestep_encoder.
    """
    policy_backend.requires_grad_(False)

    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError(
            "train_flow_partial_dense=True but policy_backend.flow does not exist."
        )
    dit = getattr(flow, "DiT", None)
    blocks = getattr(dit, "transformer_blocks", None) if dit is not None else None
    if blocks is None:
        raise RuntimeError(
            "train_flow_partial_dense=True requires flow.DiT.transformer_blocks."
        )

    raw_indices = freeze_policy.flow_partial_dense_layer_indices
    if raw_indices is None or len(raw_indices) == 0:
        raise RuntimeError(
            "train_flow_partial_dense=True requires non-empty flow_partial_dense_layer_indices."
        )
    indices = tuple(int(x) for x in raw_indices)
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"Duplicate partial-dense Flow layers: {indices}")
    bad = [idx for idx in indices if idx < 0 or idx >= len(blocks)]
    if bad:
        raise RuntimeError(
            f"Partial-dense Flow layer indices out of range: {bad}; valid=[0,{len(blocks)-1}]"
        )

    if freeze_policy.train_flow_interface_dense and freeze_policy.train_flow_interface_lora:
        raise RuntimeError("Flow-interface Dense and LoRA modes are mutually exclusive.")

    # Full-rank adaptation of the selected original DiT blocks.
    for idx in indices:
        blocks[idx].requires_grad_(True)

    interface_dense_names = []
    interface_lora_summary = None

    if freeze_policy.train_flow_interface_dense:
        for name, module in _flow_interface_dense_modules(flow):
            module.requires_grad_(True)
            interface_dense_names.append(name)

    elif freeze_policy.train_flow_interface_lora:
        interface_lora_summary = inject_flow_interface_lora(
            flow,
            rank=freeze_policy.flow_lora_rank,
            alpha=freeze_policy.flow_lora_alpha,
            dropout=freeze_policy.flow_lora_dropout,
            target_enc_vlm=freeze_policy.flow_interface_target_enc_vlm,
            target_action_encoder=freeze_policy.flow_interface_target_action_encoder,
            target_action_decoder=freeze_policy.flow_interface_target_action_decoder,
            target_state_encoder=freeze_policy.flow_interface_target_state_encoder,
            target_output=freeze_policy.flow_interface_target_output,
            target_timestep=freeze_policy.flow_interface_target_timestep,
        )
        # Injection freezes Base weights of wrapped interface modules. Keep the
        # selected dense blocks fully trainable and enable only interface LoRA tensors.
        for idx in indices:
            blocks[idx].requires_grad_(True)
        for name, param in flow.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                param.requires_grad_(True)

    trainable = [
        (name, p) for name, p in policy_backend.named_parameters() if p.requires_grad
    ]
    if not trainable:
        raise RuntimeError("Partial-dense Flow mode has no trainable parameters.")

    block_prefixes = tuple(f"flow.DiT.transformer_blocks.{idx}." for idx in indices)
    dense_interface_prefixes = (
        "flow.enc_vlm.",
        "flow.action_encoder.",
        "flow.action_decoder.",
        "flow.enc_state.",
        "flow.DiT.proj_out_1.",
        "flow.DiT.proj_out_2.",
        "flow.DiT.timestep_encoder.",
    )
    lora_interface_prefixes = dense_interface_prefixes

    unexpected = []
    for name, _ in trainable:
        if name.startswith(block_prefixes):
            continue
        if freeze_policy.train_flow_interface_dense and name.startswith(dense_interface_prefixes):
            continue
        if freeze_policy.train_flow_interface_lora and name.startswith(lora_interface_prefixes) and (
            name.endswith(".lora_A") or name.endswith(".lora_B")
        ):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            "Partial-dense/interface isolation failed; unexpected trainable params: "
            f"{unexpected[:50]}"
        )

    per_layer = {}
    for idx in indices:
        prefix = f"flow.DiT.transformer_blocks.{idx}."
        ps = [(n, p) for n, p in trainable if n.startswith(prefix)]
        if not ps:
            raise RuntimeError(f"Requested Flow layer {idx} has no trainable parameters.")
        per_layer[idx] = (len(ps), sum(p.numel() for _, p in ps))

    dense_block_params = sum(
        p.numel() for n, p in trainable if n.startswith(block_prefixes)
    )
    interface_dense_params = sum(
        p.numel() for n, p in trainable
        if freeze_policy.train_flow_interface_dense and n.startswith(dense_interface_prefixes)
    )
    interface_lora_params = sum(
        p.numel() for n, p in trainable
        if freeze_policy.train_flow_interface_lora and (
            n.endswith(".lora_A") or n.endswith(".lora_B")
        )
    )
    total = sum(p.numel() for _, p in trainable)

    if interface_lora_summary is not None and interface_lora_params != interface_lora_summary.trainable_params:
        raise RuntimeError(
            f"Interface LoRA parameter mismatch: runtime={interface_lora_params}, "
            f"summary={interface_lora_summary.trainable_params}"
        )

    interface_mode = (
        "dense" if freeze_policy.train_flow_interface_dense
        else "lora" if freeze_policy.train_flow_interface_lora
        else "none"
    )
    print(
        "[FLOW-PartialDense] isolation active: "
        f"layers={indices}, interface_mode={interface_mode}, "
        f"trainable_tensors={len(trainable)}, trainable_params={total:,}; "
        f"dense_block_params={dense_block_params:,}, "
        f"interface_dense_params={interface_dense_params:,}, "
        f"interface_lora_params={interface_lora_params:,}; "
        f"per_layer={per_layer}; "
        f"interface_dense_modules={interface_dense_names}; "
        f"interface_lora_targets={None if interface_lora_summary is None else interface_lora_summary.target_modules}; "
        "ALL upstream parameters frozen; NO residual expert is used."
    )



def _apply_vlm_flow_lora_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """Joint task LoRA: selected VLM path + all original Flow DiT blocks.

    Action-side target is intentionally FIXED across the ablation:
      - all 16 original DiT blocks
      - Attention Q/K/V/O + FFN input/output
      - NO flow.enc_vlm, output head, action encoder/decoder, AdaNorm, timestep

    Only the VLM target group changes between variants.
    """
    policy_backend.requires_grad_(False)

    flow = getattr(policy_backend, "flow", None)
    vlm = getattr(policy_backend, "vlm", None)
    if flow is None or vlm is None:
        raise RuntimeError("Joint VLM+Flow LoRA requires policy_backend.flow and policy_backend.vlm")

    if not (
        freeze_policy.vlm_lora_target_text
        or freeze_policy.vlm_lora_target_vision
        or freeze_policy.vlm_lora_target_merger
    ):
        raise RuntimeError("train_vlm_flow_lora=True requires at least one VLM LoRA target group")

    flow_summary = inject_flow_conditioning_lora(
        flow,
        rank=freeze_policy.flow_lora_rank,
        alpha=freeze_policy.flow_lora_alpha,
        dropout=freeze_policy.flow_lora_dropout,
        target_attention=True,
        target_ffn=True,
        target_enc_vlm=False,
        target_output=False,
        target_adanorm=False,
        target_timestep=False,
    )
    # Current Flow architecture: 16 blocks * (4 attention + 2 FFN) = 96.
    if flow_summary.attention_targets != 64 or flow_summary.ffn_targets != 32:
        raise RuntimeError(
            "Unexpected Action-Flow LoRA target structure: "
            f"attention={flow_summary.attention_targets}, ffn={flow_summary.ffn_targets}"
        )
    if flow_summary.enc_vlm_targets or flow_summary.output_targets or flow_summary.adanorm_targets or flow_summary.timestep_targets:
        raise RuntimeError("Action-side ablation leaked outside the 16 original DiT blocks")

    vlm_summary = inject_vlm_lora(
        vlm,
        rank=freeze_policy.vlm_lora_rank,
        alpha=freeze_policy.vlm_lora_alpha,
        dropout=freeze_policy.vlm_lora_dropout,
        target_text=freeze_policy.vlm_lora_target_text,
        text_last_n=freeze_policy.vlm_lora_text_last_n,
        target_vision=freeze_policy.vlm_lora_target_vision,
        target_merger=freeze_policy.vlm_lora_target_merger,
    )

    # Final trainability authority: ONLY LoRA A/B under VLM or Flow.
    policy_backend.requires_grad_(False)
    for root in (flow, vlm):
        for name, p in root.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    unexpected = []
    for name, _ in trainable:
        if not (name.endswith(".lora_A") or name.endswith(".lora_B")):
            unexpected.append(name)
            continue
        if not (name.startswith("flow.") or name.startswith("vlm.")):
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"Joint VLM+Flow LoRA isolation failed: {unexpected[:50]}")

    flow_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("flow.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))
    )
    vlm_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("vlm.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))
    )
    total = sum(p.numel() for _, p in trainable)
    if flow_params != flow_summary.trainable_params:
        raise RuntimeError(f"Flow LoRA param mismatch: runtime={flow_params}, summary={flow_summary.trainable_params}")
    if vlm_params != vlm_summary.trainable_params:
        raise RuntimeError(f"VLM LoRA param mismatch: runtime={vlm_params}, summary={vlm_summary.trainable_params}")

    print(
        "[VLM+FLOW-LoRA] isolation active: "
        f"total_trainable={total:,}, flow_lora={flow_params:,}, vlm_lora={vlm_params:,}; "
        f"flow_targets={len(flow_summary.target_modules)} "
        f"(attention={flow_summary.attention_targets}, ffn={flow_summary.ffn_targets}); "
        f"vlm_targets={len(vlm_summary.target_modules)} "
        f"(text={vlm_summary.text_targets}, vision={vlm_summary.vision_targets}, merger={vlm_summary.merger_targets}); "
        f"text_layers_selected={vlm_summary.text_layers_selected}, "
        f"vision_blocks={vlm_summary.vision_blocks_total}, mergers={vlm_summary.merger_modules_total}; "
        f"ranks=(flow={freeze_policy.flow_lora_rank}, vlm={freeze_policy.vlm_lora_rank}); "
        "ALL original Base parameters are frozen."
    )


def _inject_text_only_vlm_lora(policy_backend, freeze_policy: LatentWorldPolicyFreezeConfig):
    vlm = getattr(policy_backend, "vlm", None)
    if vlm is None:
        raise RuntimeError("Text-LoRA hybrid requires policy_backend.vlm")
    return inject_vlm_lora(
        vlm,
        rank=freeze_policy.vlm_lora_rank,
        alpha=freeze_policy.vlm_lora_alpha,
        dropout=freeze_policy.vlm_lora_dropout,
        target_text=True,
        text_last_n=freeze_policy.vlm_lora_text_last_n,
        target_vision=False,
        target_merger=False,
    )


def _apply_vlm_text_lora_partial_dense_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """E1: full-rank FT selected original DiT blocks + VLM text LoRA."""
    policy_backend.requires_grad_(False)
    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError("E1 requires policy_backend.flow")
    dit = getattr(flow, "DiT", None)
    blocks = getattr(dit, "transformer_blocks", None) if dit is not None else None
    if blocks is None:
        raise RuntimeError("E1 requires flow.DiT.transformer_blocks")
    indices = freeze_policy.flow_partial_dense_layer_indices
    if indices is None or len(indices) == 0:
        raise RuntimeError("E1 requires flow_partial_dense_layer_indices")
    indices = tuple(int(i) for i in indices)
    bad = [i for i in indices if i < 0 or i >= len(blocks)]
    if bad or len(indices) != len(set(indices)):
        raise RuntimeError(f"Invalid E1 dense layers: {indices}")

    vlm_summary = _inject_text_only_vlm_lora(policy_backend, freeze_policy)

    # Final authority: selected original DiT blocks + VLM LoRA A/B only.
    policy_backend.requires_grad_(False)
    for idx in indices:
        blocks[idx].requires_grad_(True)
    for name, p in policy_backend.vlm.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    block_prefixes = tuple(f"flow.DiT.transformer_blocks.{idx}." for idx in indices)
    unexpected = []
    for name, _ in trainable:
        if name.startswith(block_prefixes):
            continue
        if name.startswith("vlm.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"E1 isolation failed; unexpected trainable params: {unexpected[:50]}")

    dense_params = sum(p.numel() for n, p in trainable if n.startswith(block_prefixes))
    vlm_lora_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("vlm.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))
    )
    if vlm_lora_params != vlm_summary.trainable_params:
        raise RuntimeError(
            f"E1 VLM LoRA param mismatch: runtime={vlm_lora_params}, summary={vlm_summary.trainable_params}"
        )
    print(
        "[E1-Last8Dense+TextLoRA] isolation active: "
        f"layers={indices}, dense_params={dense_params:,}, "
        f"text_lora_params={vlm_lora_params:,}, total={dense_params + vlm_lora_params:,}, "
        f"rank={freeze_policy.vlm_lora_rank}; ALL other Base parameters frozen."
    )


def _conditioning_adapter_parameters(flow):
    items = []
    enc = getattr(flow, "enc_vlm", None)
    adapter = getattr(enc, "conditioning_adapter", None) if enc is not None else None
    if adapter is not None:
        for n, p in adapter.named_parameters():
            items.append((f"enc_vlm.conditioning_adapter.{n}", p))

    dit = getattr(flow, "DiT", None)
    blocks = getattr(dit, "transformer_blocks", None) if dit is not None else None
    if blocks is not None:
        for idx, block in enumerate(blocks):
            norm1 = getattr(block, "norm1", None)
            adapter = getattr(norm1, "conditioning_adapter", None) if norm1 is not None else None
            if adapter is not None:
                for n, p in adapter.named_parameters():
                    items.append((f"DiT.transformer_blocks.{idx}.norm1.conditioning_adapter.{n}", p))
    return items


def _apply_vlm_text_lora_conditioning_adapter_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """E7: VLM text LoRA + CLARE-style enc_vlm/AdaLN nonlinear side branches."""
    policy_backend.requires_grad_(False)
    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError("E7 requires policy_backend.flow")

    adapter_items = _conditioning_adapter_parameters(flow)
    if not adapter_items:
        raise RuntimeError(
            "E7 requested but no conditioning adapters exist. Set "
            "framework.action_model.flow_cfg.conditioning_adapter_bottleneck > 0."
        )
    vlm_summary = _inject_text_only_vlm_lora(policy_backend, freeze_policy)

    # Final authority: only new conditioning adapters + VLM LoRA A/B.
    policy_backend.requires_grad_(False)
    for _, p in _conditioning_adapter_parameters(flow):
        p.requires_grad_(True)
    for name, p in policy_backend.vlm.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    unexpected = []
    for name, _ in trainable:
        if ".conditioning_adapter." in name and name.startswith("flow."):
            continue
        if name.startswith("vlm.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"E7 isolation failed; unexpected trainable params: {unexpected[:50]}")

    adapter_params = sum(
        p.numel() for n, p in trainable if n.startswith("flow.") and ".conditioning_adapter." in n
    )
    vlm_lora_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("vlm.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))
    )
    if vlm_lora_params != vlm_summary.trainable_params:
        raise RuntimeError(
            f"E7 VLM LoRA param mismatch: runtime={vlm_lora_params}, summary={vlm_summary.trainable_params}"
        )
    print(
        "[E7-TextLoRA+ConditioningAdapter] isolation active: "
        f"adapter_params={adapter_params:,}, text_lora_params={vlm_lora_params:,}, "
        f"total={adapter_params + vlm_lora_params:,}, rank={freeze_policy.vlm_lora_rank}; "
        "ALL original Base VLM/LaWM/Flow weights frozen."
    )


def _apply_partial_dense_conditioning_adapter_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
    *,
    with_text_lora: bool,
) -> None:
    """Selected original DiT blocks dense FT + CLARE-style conditioning adapters.

    If ``with_text_lora`` is True, the retained VLM language layers additionally
    receive text-only LoRA.  All other Base VLM/LaWM/Flow parameters remain
    frozen.  Conditioning adapters are new nonlinear side branches in enc_vlm
    and all AdaLN modules and are therefore task-specific parameters.
    """
    policy_backend.requires_grad_(False)
    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError("Partial-dense + conditioning mode requires policy_backend.flow")
    dit = getattr(flow, "DiT", None)
    blocks = getattr(dit, "transformer_blocks", None) if dit is not None else None
    if blocks is None:
        raise RuntimeError("Partial-dense + conditioning mode requires flow.DiT.transformer_blocks")

    indices = freeze_policy.flow_partial_dense_layer_indices
    if indices is None or len(indices) == 0:
        raise RuntimeError("Partial-dense + conditioning mode requires flow_partial_dense_layer_indices")
    indices = tuple(int(i) for i in indices)
    bad = [i for i in indices if i < 0 or i >= len(blocks)]
    if bad or len(indices) != len(set(indices)):
        raise RuntimeError(f"Invalid dense layers: {indices}")

    adapter_items = _conditioning_adapter_parameters(flow)
    if not adapter_items:
        raise RuntimeError(
            "Conditioning mode requested but no conditioning adapters exist. Set "
            "framework.action_model.flow_cfg.conditioning_adapter_bottleneck > 0."
        )

    vlm_summary = None
    if with_text_lora:
        vlm_summary = _inject_text_only_vlm_lora(policy_backend, freeze_policy)

    # Final trainability authority.
    policy_backend.requires_grad_(False)
    for idx in indices:
        blocks[idx].requires_grad_(True)
    # Make ALL conditioning adapters trainable, including those inside frozen blocks.
    for _, p in _conditioning_adapter_parameters(flow):
        p.requires_grad_(True)
    if with_text_lora:
        for name, p in policy_backend.vlm.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    block_prefixes = tuple(f"flow.DiT.transformer_blocks.{idx}." for idx in indices)
    unexpected = []
    for name, _ in trainable:
        if name.startswith(block_prefixes) and ".conditioning_adapter." not in name:
            continue
        if name.startswith("flow.") and ".conditioning_adapter." in name:
            continue
        if with_text_lora and name.startswith("vlm.") and (
            name.endswith(".lora_A") or name.endswith(".lora_B")
        ):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            "Partial-dense + conditioning isolation failed; unexpected trainable params: "
            f"{unexpected[:50]}"
        )

    dense_params = sum(
        p.numel() for n, p in trainable
        if n.startswith(block_prefixes) and ".conditioning_adapter." not in n
    )
    adapter_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("flow.") and ".conditioning_adapter." in n
    )
    vlm_lora_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("vlm.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))
    )
    if with_text_lora:
        assert vlm_summary is not None
        if vlm_lora_params != vlm_summary.trainable_params:
            raise RuntimeError(
                "VLM LoRA param mismatch: "
                f"runtime={vlm_lora_params}, summary={vlm_summary.trainable_params}"
            )
    elif vlm_lora_params != 0:
        raise RuntimeError("Action-only conditioning mode unexpectedly contains VLM LoRA params")

    print(
        "[PARTIAL-DENSE+COND] isolation active: "
        f"layers={indices}, dense_params={dense_params:,}, adapter_params={adapter_params:,}, "
        f"text_lora_params={vlm_lora_params:,}, total={dense_params + adapter_params + vlm_lora_params:,}, "
        f"with_text_lora={with_text_lora}; ALL other Base parameters frozen."
    )



def _dit_nonlinear_adapter_parameters(flow):
    """Return task-specific activation-space DiT side-adapter parameters."""
    items = []
    dit = getattr(flow, "DiT", None)
    blocks = getattr(dit, "transformer_blocks", None) if dit is not None else None
    if blocks is None:
        return items
    for idx, block in enumerate(blocks):
        for attr in ("attn_nonlinear_adapter", "ffn_nonlinear_adapter"):
            adapter = getattr(block, attr, None)
            if adapter is None:
                continue
            for n, p in adapter.named_parameters():
                items.append((f"DiT.transformer_blocks.{idx}.{attr}.{n}", p))
    return items


def _apply_partial_dense_conditioning_nonlinear_adapter_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
    *,
    with_text_lora: bool,
) -> None:
    """Dense selected Base DiT blocks + conditioning + nonlinear side adapters.

    The nonlinear adapters are expected to be instantiated only in the DiT layers
    configured by ``flow_cfg.dit_nonlinear_adapter_layer_indices``.  In the v8
    protocol these are the still-frozen DiT layers, so dense and nonlinear
    plasticity do not overlap.
    """
    policy_backend.requires_grad_(False)
    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError("v8 partial-dense nonlinear mode requires policy_backend.flow")
    dit = getattr(flow, "DiT", None)
    blocks = getattr(dit, "transformer_blocks", None) if dit is not None else None
    if blocks is None:
        raise RuntimeError("v8 partial-dense nonlinear mode requires flow.DiT.transformer_blocks")

    indices = freeze_policy.flow_partial_dense_layer_indices
    if indices is None or len(indices) == 0:
        raise RuntimeError("v8 partial-dense nonlinear mode requires flow_partial_dense_layer_indices")
    indices = tuple(int(i) for i in indices)
    bad = [i for i in indices if i < 0 or i >= len(blocks)]
    if bad or len(indices) != len(set(indices)):
        raise RuntimeError(f"Invalid dense layers: {indices}")

    cond_items = _conditioning_adapter_parameters(flow)
    nl_items = _dit_nonlinear_adapter_parameters(flow)
    if not cond_items:
        raise RuntimeError(
            "v8 mode requested but no conditioning adapters exist. Set "
            "framework.action_model.flow_cfg.conditioning_adapter_bottleneck > 0."
        )
    if not nl_items:
        raise RuntimeError(
            "v8 mode requested but no DiT nonlinear adapters exist. Set "
            "framework.action_model.flow_cfg.dit_nonlinear_adapter_bottleneck > 0."
        )

    vlm_summary = None
    if with_text_lora:
        vlm_summary = _inject_text_only_vlm_lora(policy_backend, freeze_policy)

    # Final trainability authority.
    policy_backend.requires_grad_(False)
    for idx in indices:
        blocks[idx].requires_grad_(True)
    for _, p in _conditioning_adapter_parameters(flow):
        p.requires_grad_(True)
    for _, p in _dit_nonlinear_adapter_parameters(flow):
        p.requires_grad_(True)
    if with_text_lora:
        for name, p in policy_backend.vlm.named_parameters():
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    block_prefixes = tuple(f"flow.DiT.transformer_blocks.{idx}." for idx in indices)
    unexpected = []
    for name, _ in trainable:
        if name.startswith(block_prefixes) and (
            ".attn_nonlinear_adapter." not in name and ".ffn_nonlinear_adapter." not in name
            and ".conditioning_adapter." not in name
        ):
            continue
        if name.startswith("flow.") and ".conditioning_adapter." in name:
            continue
        if name.startswith("flow.") and (
            ".attn_nonlinear_adapter." in name or ".ffn_nonlinear_adapter." in name
        ):
            continue
        if with_text_lora and name.startswith("vlm.") and (
            name.endswith(".lora_A") or name.endswith(".lora_B")
        ):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            "v8 partial-dense + conditioning + nonlinear isolation failed; "
            f"unexpected trainable params: {unexpected[:50]}"
        )

    dense_params = sum(
        p.numel() for n, p in trainable
        if n.startswith(block_prefixes)
        and ".conditioning_adapter." not in n
        and ".attn_nonlinear_adapter." not in n
        and ".ffn_nonlinear_adapter." not in n
    )
    cond_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("flow.") and ".conditioning_adapter." in n
    )
    nl_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("flow.") and (
            ".attn_nonlinear_adapter." in n or ".ffn_nonlinear_adapter." in n
        )
    )
    vlm_lora_params = sum(
        p.numel() for n, p in trainable
        if n.startswith("vlm.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))
    )
    if with_text_lora:
        assert vlm_summary is not None
        if vlm_lora_params != vlm_summary.trainable_params:
            raise RuntimeError(
                "VLM LoRA param mismatch: "
                f"runtime={vlm_lora_params}, summary={vlm_summary.trainable_params}"
            )
    elif vlm_lora_params != 0:
        raise RuntimeError("Action-only v8 mode unexpectedly contains VLM LoRA params")

    print(
        "[V8-PARTIAL-DENSE+COND+NONLINEAR] isolation active: "
        f"dense_layers={indices}, dense_params={dense_params:,}, conditioning_params={cond_params:,}, "
        f"nonlinear_params={nl_params:,}, text_lora_params={vlm_lora_params:,}, "
        f"total={dense_params + cond_params + nl_params + vlm_lora_params:,}, "
        f"with_text_lora={with_text_lora}; ALL other Base parameters frozen."
    )


def _apply_conditioning_nonlinear_adapter_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    """Action-only conditioning + DiT nonlinear side adapters, no dense Base FT."""
    policy_backend.requires_grad_(False)
    flow = getattr(policy_backend, "flow", None)
    if flow is None:
        raise RuntimeError("v8 conditioning+nonlinear mode requires policy_backend.flow")
    cond_items = _conditioning_adapter_parameters(flow)
    nl_items = _dit_nonlinear_adapter_parameters(flow)
    if not cond_items:
        raise RuntimeError("No conditioning adapters exist for v8 B4")
    if not nl_items:
        raise RuntimeError("No DiT nonlinear adapters exist for v8 B4")

    policy_backend.requires_grad_(False)
    for _, p in cond_items:
        p.requires_grad_(True)
    for _, p in nl_items:
        p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    unexpected = []
    for name, _ in trainable:
        if name.startswith("flow.") and ".conditioning_adapter." in name:
            continue
        if name.startswith("flow.") and (
            ".attn_nonlinear_adapter." in name or ".ffn_nonlinear_adapter." in name
        ):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            "v8 conditioning+nonlinear isolation failed; unexpected trainable params: "
            f"{unexpected[:50]}"
        )
    cond_params = sum(
        p.numel() for n, p in trainable if ".conditioning_adapter." in n
    )
    nl_params = sum(
        p.numel() for n, p in trainable
        if ".attn_nonlinear_adapter." in n or ".ffn_nonlinear_adapter." in n
    )
    print(
        "[V8-COND+NONLINEAR] isolation active: "
        f"conditioning_params={cond_params:,}, nonlinear_params={nl_params:,}, "
        f"total={cond_params + nl_params:,}; ALL original Base parameters frozen."
    )


def _inject_routing_v2_qformer_lawm_lora(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
):
    qformer = getattr(policy_backend, "vlm_to_lam", None)
    if qformer is None:
        raise RuntimeError("Routing-V2 requires policy_backend.vlm_to_lam")
    lam = getattr(policy_backend, "lam", None)
    decoder = getattr(lam, "decoder", None) if lam is not None else None
    if decoder is None:
        raise RuntimeError("Routing-V2 requires policy_backend.lam.decoder")

    q_summary = inject_recursive_linear_lora(
        qformer,
        rank=freeze_policy.routing_v2_qformer_lora_rank,
        alpha=freeze_policy.routing_v2_qformer_lora_alpha,
        dropout=freeze_policy.routing_v2_qformer_lora_dropout,
        module_name="VLMToLAM/QFormer",
    )
    w_summary = inject_recursive_linear_lora(
        decoder,
        rank=freeze_policy.routing_v2_lawm_lora_rank,
        alpha=freeze_policy.routing_v2_lawm_lora_alpha,
        dropout=freeze_policy.routing_v2_lawm_lora_dropout,
        module_name="LaWM decoder",
    )
    return q_summary, w_summary


def _routing_v2_query_delta_parameters(policy_backend):
    items = []
    for name in ("routing_v2_act_query_delta", "routing_v2_flow_query_delta"):
        p = getattr(policy_backend, name, None)
        if p is not None:
            items.append((name, p))
    return items


def _apply_routing_v2_skill_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    if not bool(freeze_policy.vlm_lora_target_text):
        raise RuntimeError("Routing-V2 skill requires vlm_lora_target_text=true")
    if bool(freeze_policy.vlm_lora_target_vision) or bool(freeze_policy.vlm_lora_target_merger):
        raise RuntimeError("Routing-V2 V1 skill keeps VLM adaptation text-only")
    if not bool(freeze_policy.train_flow_partial_dense):
        raise RuntimeError("Routing-V2 skill reuses B2 and requires train_flow_partial_dense=true")

    # Reuse the already-validated B2 action expert implementation.
    _apply_partial_dense_conditioning_nonlinear_adapter_training(
        policy_backend, freeze_policy, with_text_lora=True
    )
    q_summary, w_summary = _inject_routing_v2_qformer_lawm_lora(
        policy_backend, freeze_policy
    )
    q_lora = set_lora_trainable(policy_backend.vlm_to_lam, True)
    w_lora = set_lora_trainable(policy_backend.lam.decoder, True)

    query_items = _routing_v2_query_delta_parameters(policy_backend)
    if len(query_items) != 2:
        raise RuntimeError(
            "Routing-V2 skill requires zero-initialized act/flow query deltas; "
            "set framework.action_model.routing_v2_enable_query_delta=true"
        )
    for _, p in query_items:
        p.requires_grad_(True)

    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    def allowed(name: str) -> bool:
        if name.startswith("vlm.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            return True
        if name.startswith("vlm_to_lam.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            return True
        if name.startswith("lam.decoder.") and (name.endswith(".lora_A") or name.endswith(".lora_B")):
            return True
        if name in {"routing_v2_act_query_delta", "routing_v2_flow_query_delta"}:
            return True
        if name.startswith("flow."):
            # Existing B2 authority already audited the exact Flow subset.
            return True
        return False
    unexpected = [n for n, _ in trainable if not allowed(n)]
    if unexpected:
        raise RuntimeError(
            "Routing-V2 skill isolation failed; unexpected trainable params: "
            f"{unexpected[:50]}"
        )

    groups = {
        "flow_action_skill": sum(p.numel() for n, p in trainable if n.startswith("flow.")),
        "vlm_text_lora": sum(p.numel() for n, p in trainable if n.startswith("vlm.") and (n.endswith(".lora_A") or n.endswith(".lora_B"))),
        "query_delta": sum(p.numel() for n, p in query_items),
        "qformer_lora": q_lora,
        "lawm_lora": w_lora,
    }
    groups["total"] = sum(groups.values())
    print(
        "[RoutingV2][SKILL][PARAMS] "
        + ", ".join(f"{k}={v:,}" for k, v in groups.items())
    )
    print(
        "[RoutingV2][SKILL][LORA] "
        f"qformer_targets={len(q_summary.target_modules)} params={q_summary.trainable_params:,}; "
        f"lawm_targets={len(w_summary.target_modules)} params={w_summary.trainable_params:,}"
    )
    print(f"[RoutingV2][SKILL][LORA] qformer_modules={q_summary.target_modules}")
    if q_summary.skipped_modules:
        print(
            "[RoutingV2][SKILL][LORA][SKIP] QFormer modules bypassed by functional MHA forward: "
            f"{q_summary.skipped_modules}"
        )
    print(f"[RoutingV2][SKILL][LORA] lawm_modules={w_summary.target_modules}")
    if w_summary.skipped_modules:
        print(
            "[RoutingV2][SKILL][LORA][SKIP] LaWM modules bypassed by functional MHA forward: "
            f"{w_summary.skipped_modules}"
        )


def _apply_routing_v2_memory_training(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    # Recreate all skill-path structural LoRA modules before checkpoint load, but
    # keep them frozen. Flow side adapters are created by flow_cfg at model init.
    if not bool(freeze_policy.vlm_lora_target_text):
        raise RuntimeError("Routing-V2 memory phase requires VLM text LoRA structure")
    _inject_text_only_vlm_lora(policy_backend, freeze_policy)
    q_summary, w_summary = _inject_routing_v2_qformer_lawm_lora(
        policy_backend, freeze_policy
    )
    policy_backend.requires_grad_(False)
    memory = getattr(policy_backend, "routing_v2_memory", None)
    if memory is None:
        raise RuntimeError(
            "Routing-V2 memory phase requires framework.action_model.routing_v2_enable_memory=true"
        )
    memory.requires_grad_(True)
    trainable = [(n, p) for n, p in policy_backend.named_parameters() if p.requires_grad]
    unexpected = [n for n, _ in trainable if not n.startswith("routing_v2_memory.")]
    if unexpected:
        raise RuntimeError(
            "Routing-V2 memory isolation failed; only routing_v2_memory may train: "
            f"{unexpected[:50]}"
        )
    mem_params = sum(p.numel() for _, p in trainable)
    print(
        "[RoutingV2][MEMORY][PARAMS] "
        f"trainable={mem_params:,}; summary={memory.parameter_summary()}; "
        f"skill_struct_qformer_lora={q_summary.trainable_params:,}; "
        f"skill_struct_lawm_lora={w_summary.trainable_params:,}; all skill params FROZEN"
    )


def _enable_expert_latent_head_training(policy_backend) -> None:
    """Unfreeze only the Routing-V1 z* head after any isolation mode froze the Base model."""
    flow = getattr(policy_backend, "flow", None)
    head = getattr(flow, "expert_latent_head", None) if flow is not None else None
    if head is None:
        raise RuntimeError(
            "train_expert_latent_head=True but policy_backend.flow.expert_latent_head does not exist. "
            "Set framework.action_model.flow_cfg.enable_expert_latent_head=true."
        )
    head.requires_grad_(True)
    params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    if params <= 0:
        raise RuntimeError("Expert latent head was requested trainable but has zero trainable parameters.")
    print(f"[freeze-policy] Routing-V1 expert latent head trainable: params={params:,}.")


def apply_policy_freeze(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    freeze_qwen3vl(
        policy_backend.vlm,
        freeze_vision_backbone=freeze_policy.freeze_vision_backbone,
        freeze_llm_backbone=freeze_policy.freeze_llm_backbone,
        freeze_last_llm_layer=freeze_policy.freeze_last_llm_layer,
        freeze_embedding=freeze_policy.freeze_embedding,
        unfreeze_vision_merger=freeze_policy.unfreeze_vision_merger,
    )

    llm_module = None
    if (
        freeze_policy.keep_llm_first_n_layers is not None
        or (
            freeze_policy.freeze_llm_backbone
            and freeze_policy.unfreeze_llm_last_n_layers is not None
            and freeze_policy.unfreeze_llm_last_n_layers > 0
        )
    ):
        llm_module = _resolve_llm_module(policy_backend.vlm)

    if freeze_policy.keep_llm_first_n_layers is not None:
        _keep_first_n_llm_layers(
            llm_module,
            freeze_policy.keep_llm_first_n_layers,
        )

    if (
        freeze_policy.freeze_llm_backbone
        and freeze_policy.unfreeze_llm_last_n_layers is not None
        and freeze_policy.unfreeze_llm_last_n_layers > 0
    ):
        _unfreeze_last_n_llm_layers(
            llm_module,
            freeze_policy.unfreeze_llm_last_n_layers,
        )

    _apply_strict_vlm_interface_freeze(
        policy_backend,
        freeze_policy,
    )

    for p in policy_backend.lam.parameters():
        p.requires_grad = False

    if freeze_policy.unfreeze_lam_decoder:
        lam_decoder = getattr(policy_backend.lam, "decoder", None)
        if lam_decoder is not None:
            for p in lam_decoder.parameters():
                p.requires_grad = True

    if freeze_policy.train_routing_v2_skill and freeze_policy.train_routing_v2_memory_only:
        raise RuntimeError("Routing-V2 skill and memory modes are mutually exclusive.")
    if freeze_policy.train_routing_v2_skill:
        _apply_routing_v2_skill_training(policy_backend, freeze_policy)
        return
    if freeze_policy.train_routing_v2_memory_only:
        _apply_routing_v2_memory_training(policy_backend, freeze_policy)
        return

    hybrid_modes = [
        freeze_policy.train_vlm_flow_lora,
        freeze_policy.train_vlm_text_lora_partial_dense,
        freeze_policy.train_vlm_text_lora_conditioning_adapter,
        freeze_policy.train_flow_partial_dense_conditioning_adapter,
        freeze_policy.train_vlm_text_lora_partial_dense_conditioning_adapter,
        freeze_policy.train_flow_partial_dense_conditioning_nonlinear_adapter,
        freeze_policy.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter,
        freeze_policy.train_flow_conditioning_nonlinear_adapter,
    ]
    if sum(bool(x) for x in hybrid_modes) > 1:
        raise RuntimeError(
            "VLM/conditioning hybrid training modes are mutually exclusive."
        )

    if freeze_policy.train_vlm_flow_lora and (
        freeze_policy.train_flow_only
        or freeze_policy.train_flow_lora
        or freeze_policy.train_flow_residual_expert
        or freeze_policy.train_flow_partial_dense
        or freeze_policy.train_flow_interface_dense
        or freeze_policy.train_flow_interface_lora
    ):
        raise RuntimeError(
            "train_vlm_flow_lora is a standalone diagnostic mode and cannot be combined "
            "with other Flow adaptation modes."
        )

    if freeze_policy.train_vlm_text_lora_partial_dense:
        if not freeze_policy.train_flow_partial_dense:
            raise RuntimeError("E1 requires train_flow_partial_dense=True")
        if freeze_policy.train_flow_interface_dense or freeze_policy.train_flow_interface_lora:
            raise RuntimeError("E1 keeps Flow interfaces frozen")
        if freeze_policy.train_flow_only or freeze_policy.train_flow_lora or freeze_policy.train_flow_residual_expert:
            raise RuntimeError("E1 cannot combine with other Flow adaptation modes")

    if freeze_policy.train_vlm_text_lora_conditioning_adapter:
        if (freeze_policy.train_flow_only or freeze_policy.train_flow_lora or
            freeze_policy.train_flow_residual_expert or freeze_policy.train_flow_partial_dense or
            freeze_policy.train_flow_interface_dense or freeze_policy.train_flow_interface_lora):
            raise RuntimeError("E7 is standalone apart from VLM text LoRA + conditioning adapters")

    if freeze_policy.train_flow_partial_dense_conditioning_adapter:
        if not freeze_policy.train_flow_partial_dense:
            raise RuntimeError("Action-only partial-dense + conditioning requires train_flow_partial_dense=True")
        if (freeze_policy.train_flow_only or freeze_policy.train_flow_lora or
            freeze_policy.train_flow_residual_expert or freeze_policy.train_flow_interface_dense or
            freeze_policy.train_flow_interface_lora):
            raise RuntimeError("Action-only partial-dense + conditioning cannot combine with other Flow modes")

    if freeze_policy.train_vlm_text_lora_partial_dense_conditioning_adapter:
        if not freeze_policy.train_flow_partial_dense:
            raise RuntimeError("Text-LoRA + partial-dense + conditioning requires train_flow_partial_dense=True")
        if (freeze_policy.train_flow_only or freeze_policy.train_flow_lora or
            freeze_policy.train_flow_residual_expert or freeze_policy.train_flow_interface_dense or
            freeze_policy.train_flow_interface_lora):
            raise RuntimeError("Text-LoRA + partial-dense + conditioning cannot combine with other Flow modes")

    if freeze_policy.train_flow_partial_dense_conditioning_nonlinear_adapter:
        if not freeze_policy.train_flow_partial_dense:
            raise RuntimeError(
                "Action-only v8 partial-dense+conditioning+nonlinear requires train_flow_partial_dense=True"
            )
        if (freeze_policy.train_flow_only or freeze_policy.train_flow_lora or
            freeze_policy.train_flow_residual_expert or freeze_policy.train_flow_interface_dense or
            freeze_policy.train_flow_interface_lora):
            raise RuntimeError("v8 action-only partial-dense nonlinear mode cannot combine with other Flow modes")

    if freeze_policy.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter:
        if not freeze_policy.train_flow_partial_dense:
            raise RuntimeError(
                "Text-LoRA v8 partial-dense+conditioning+nonlinear requires train_flow_partial_dense=True"
            )
        if (freeze_policy.train_flow_only or freeze_policy.train_flow_lora or
            freeze_policy.train_flow_residual_expert or freeze_policy.train_flow_interface_dense or
            freeze_policy.train_flow_interface_lora):
            raise RuntimeError("v8 Text-LoRA partial-dense nonlinear mode cannot combine with other Flow modes")

    if freeze_policy.train_flow_conditioning_nonlinear_adapter:
        if (freeze_policy.train_flow_only or freeze_policy.train_flow_lora or
            freeze_policy.train_flow_residual_expert or freeze_policy.train_flow_partial_dense or
            freeze_policy.train_flow_interface_dense or freeze_policy.train_flow_interface_lora):
            raise RuntimeError("v8 conditioning+nonlinear mode is standalone and does not use dense Base FT")

    if (freeze_policy.train_flow_interface_dense or freeze_policy.train_flow_interface_lora) and not freeze_policy.train_flow_partial_dense:
        raise RuntimeError(
            "Flow-interface adaptation is defined here only as a companion to "
            "train_flow_partial_dense=True."
        )

    if freeze_policy.train_flow_interface_dense and freeze_policy.train_flow_interface_lora:
        raise RuntimeError("Flow-interface Dense and LoRA are mutually exclusive.")

    if freeze_policy.train_flow_partial_dense and (
        freeze_policy.train_flow_only
        or freeze_policy.train_flow_lora
        or freeze_policy.train_flow_residual_expert
    ):
        raise RuntimeError(
            "train_flow_partial_dense cannot be combined with train_flow_only, "
            "general Flow-LoRA, or residual-expert training.  Dedicated "
            "Flow-interface Dense/LoRA companions are allowed."
        )

    if freeze_policy.train_flow_only and (
        freeze_policy.train_flow_lora or freeze_policy.train_flow_residual_expert
    ):
        raise RuntimeError(
            "train_flow_only cannot be combined with LoRA or residual expert training."
        )

    if freeze_policy.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter:
        _apply_partial_dense_conditioning_nonlinear_adapter_training(
            policy_backend,
            freeze_policy,
            with_text_lora=True,
        )
    elif freeze_policy.train_flow_partial_dense_conditioning_nonlinear_adapter:
        _apply_partial_dense_conditioning_nonlinear_adapter_training(
            policy_backend,
            freeze_policy,
            with_text_lora=False,
        )
    elif freeze_policy.train_flow_conditioning_nonlinear_adapter:
        _apply_conditioning_nonlinear_adapter_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_vlm_text_lora_partial_dense_conditioning_adapter:
        _apply_partial_dense_conditioning_adapter_training(
            policy_backend,
            freeze_policy,
            with_text_lora=True,
        )
    elif freeze_policy.train_flow_partial_dense_conditioning_adapter:
        _apply_partial_dense_conditioning_adapter_training(
            policy_backend,
            freeze_policy,
            with_text_lora=False,
        )
    elif freeze_policy.train_vlm_text_lora_partial_dense:
        _apply_vlm_text_lora_partial_dense_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_vlm_text_lora_conditioning_adapter:
        _apply_vlm_text_lora_conditioning_adapter_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_vlm_flow_lora:
        _apply_vlm_flow_lora_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_flow_partial_dense:
        _apply_flow_partial_dense_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_flow_lora and freeze_policy.train_flow_residual_expert:
        _apply_flow_residual_plus_lora_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_flow_lora:
        _apply_flow_conditioning_lora(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_flow_residual_expert:
        _apply_flow_residual_expert_training(
            policy_backend,
            freeze_policy,
        )
    elif freeze_policy.train_flow_only:
        _apply_flow_only_training(policy_backend)

    # Apply this last because B1/B2 isolation modes intentionally call
    # policy_backend.requires_grad_(False) before reopening only their allowed
    # expert parameters.  The z* head is a new task-specific expert parameter
    # and must therefore be reopened after those isolation functions finish.
    if freeze_policy.train_expert_latent_head:
        _enable_expert_latent_head_training(policy_backend)
