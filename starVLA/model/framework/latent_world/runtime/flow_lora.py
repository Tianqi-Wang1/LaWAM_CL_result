from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowLoRALinear(nn.Linear):
    """Frozen Base Linear + trainable LoRA residual."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"LoRA dropout must be in [0,1), got {dropout}")

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = float(alpha) / float(rank)
        self.lora_dropout_p = float(dropout)
        self.lora_dropout = (
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )

        # nn.Linear weight: [out, in]
        self.lora_A = nn.Parameter(
            torch.empty(
                self.lora_rank,
                in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.empty(
                out_features,
                self.lora_rank,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        # Function preserving at step 0: B @ A == 0.
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "FlowLoRALinear":
        if isinstance(linear, FlowLoRALinear):
            if (
                linear.lora_rank != int(rank)
                or float(linear.lora_alpha) != float(alpha)
                or float(linear.lora_dropout_p) != float(dropout)
            ):
                raise RuntimeError(
                    "Existing FlowLoRALinear has a different config."
                )
            return linear

        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear, got {type(linear).__name__}"
            )

        new = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # Preserve exact Base parameters / checkpoint keys.
        new.weight = linear.weight
        new.bias = linear.bias
        new.weight.requires_grad_(False)
        if new.bias is not None:
            new.bias.requires_grad_(False)

        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        z = self.lora_dropout(x)
        z = F.linear(z, self.lora_A)
        z = F.linear(z, self.lora_B)
        return base + self.lora_scaling * z




class FlowLoRACategorySpecificLinear(nn.Module):
    """Category-specific Base Linear + shared task LoRA residual.

    The Base module stores ``W`` with shape [num_categories, in, out].  The
    LoRA branch is shared across categories because the adapter is selected at
    the task/expert level; this keeps the adapter compact while preserving the
    exact Base category-specific parameters.
    """

    def __init__(
        self,
        *,
        num_categories: int,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.num_categories = int(num_categories)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = float(alpha) / float(rank)
        self.lora_dropout_p = float(dropout)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.W = nn.Parameter(
            torch.empty(self.num_categories, self.in_features, self.out_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.empty(self.num_categories, self.out_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.lora_A = nn.Parameter(
            torch.empty(self.lora_rank, self.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.out_features, self.lora_rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_category_specific(
        cls,
        old: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "FlowLoRACategorySpecificLinear":
        if isinstance(old, cls):
            return old
        if not hasattr(old, "W") or not hasattr(old, "b"):
            raise TypeError(f"Expected CategorySpecificLinear-like module, got {type(old).__name__}")
        W = old.W
        b = old.b
        if W.ndim != 3 or b.ndim != 2:
            raise RuntimeError(
                f"Invalid category-specific shapes: W={tuple(W.shape)}, b={tuple(b.shape)}"
            )
        new = cls(
            num_categories=int(W.shape[0]),
            in_features=int(W.shape[1]),
            out_features=int(W.shape[2]),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=W.device,
            dtype=W.dtype,
        )
        new.W = W
        new.b = b
        new.W.requires_grad_(False)
        new.b.requires_grad_(False)
        return new

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        base = torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
        z = self.lora_dropout(x)
        z = F.linear(z, self.lora_A)
        z = F.linear(z, self.lora_B)
        return base + self.lora_scaling * z


@dataclass(frozen=True)
class FlowInterfaceLoRASummary:
    target_modules: tuple[str, ...]
    trainable_tensors: int
    trainable_params: int
    rank: int
    alpha: float
    dropout: float
    enc_vlm_targets: int
    action_encoder_targets: int
    action_decoder_targets: int
    state_encoder_targets: int
    output_targets: int
    timestep_targets: int


def _replace_category_specific_attr(
    parent: nn.Module,
    attr: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    old = getattr(parent, attr)
    setattr(
        parent,
        attr,
        FlowLoRACategorySpecificLinear.from_category_specific(
            old, rank=rank, alpha=alpha, dropout=dropout
        ),
    )


def inject_flow_interface_lora(
    flow: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_enc_vlm: bool = True,
    target_action_encoder: bool = True,
    target_action_decoder: bool = True,
    target_state_encoder: bool = True,
    target_output: bool = True,
    target_timestep: bool = True,
) -> FlowInterfaceLoRASummary:
    """Inject LoRA only into the Flow input/output conditioning interface.

    This deliberately excludes all DiT transformer blocks so it can be paired
    with dense Last8 adaptation without overlapping parameterizations.
    """
    targets: List[str] = []
    enc_vlm_names: List[str] = []
    action_encoder_names: List[str] = []
    action_decoder_names: List[str] = []
    state_encoder_names: List[str] = []
    output_names: List[str] = []
    timestep_names: List[str] = []

    if target_enc_vlm:
        enc_vlm = getattr(flow, "enc_vlm", None)
        if enc_vlm is None:
            raise RuntimeError("flow.enc_vlm does not exist")
        enc_vlm_names = _inject_all_linears_recursive(
            enc_vlm, prefix="enc_vlm", rank=rank, alpha=alpha, dropout=dropout
        )
        if len(enc_vlm_names) != 1:
            raise RuntimeError(f"Expected 1 enc_vlm Linear, got {enc_vlm_names}")

    if target_action_encoder:
        ae = getattr(flow, "action_encoder", None)
        if ae is None:
            raise RuntimeError("flow.action_encoder does not exist")
        for attr in ("W1", "W2"):
            _replace_category_specific_attr(ae, attr, rank=rank, alpha=alpha, dropout=dropout)
            action_encoder_names.append(f"action_encoder.{attr}")

    if target_action_decoder:
        ad = getattr(flow, "action_decoder", None)
        if ad is None:
            raise RuntimeError("flow.action_decoder does not exist")
        for attr in ("layer1", "layer2"):
            _replace_category_specific_attr(ad, attr, rank=rank, alpha=alpha, dropout=dropout)
            action_decoder_names.append(f"action_decoder.{attr}")

    if target_state_encoder:
        se = getattr(flow, "enc_state", None)
        if se is not None:
            for attr in ("W1", "W2"):
                _replace_category_specific_attr(se, attr, rank=rank, alpha=alpha, dropout=dropout)
                state_encoder_names.append(f"enc_state.{attr}")

    dit = getattr(flow, "DiT", None)
    if dit is None:
        raise RuntimeError("flow.DiT does not exist")

    if target_output:
        for attr in ("proj_out_1", "proj_out_2"):
            _replace_attr_linear(dit, attr, rank=rank, alpha=alpha, dropout=dropout)
            output_names.append(f"DiT.{attr}")

    if target_timestep:
        te = getattr(dit, "timestep_encoder", None)
        if te is None:
            raise RuntimeError("flow.DiT.timestep_encoder does not exist")
        timestep_names = _inject_all_linears_recursive(
            te, prefix="DiT.timestep_encoder", rank=rank, alpha=alpha, dropout=dropout
        )
        if len(timestep_names) != 2:
            raise RuntimeError(f"Expected 2 timestep Linear modules, got {timestep_names}")

    targets = (enc_vlm_names + action_encoder_names + action_decoder_names +
               state_encoder_names + output_names + timestep_names)
    if not targets or len(targets) != len(set(targets)):
        raise RuntimeError(f"Invalid Flow-interface LoRA target set: {targets}")

    lora_params = [
        (name, p) for name, p in flow.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    ]
    if len(lora_params) != 2 * len(targets):
        raise RuntimeError(
            f"Interface LoRA tensor mismatch: targets={len(targets)}, tensors={len(lora_params)}"
        )
    return FlowInterfaceLoRASummary(
        target_modules=tuple(targets),
        trainable_tensors=len(lora_params),
        trainable_params=sum(p.numel() for _, p in lora_params),
        rank=int(rank), alpha=float(alpha), dropout=float(dropout),
        enc_vlm_targets=len(enc_vlm_names),
        action_encoder_targets=len(action_encoder_names),
        action_decoder_targets=len(action_decoder_names),
        state_encoder_targets=len(state_encoder_names),
        output_targets=len(output_names),
        timestep_targets=len(timestep_names),
    )


@dataclass(frozen=True)
class FlowConditioningLoRASummary:
    num_blocks: int
    target_modules: tuple[str, ...]
    trainable_tensors: int
    trainable_params: int
    rank: int
    alpha: float
    dropout: float

    target_adanorm: bool
    target_timestep: bool

    attention_targets: int
    ffn_targets: int
    enc_vlm_targets: int
    output_targets: int
    adanorm_targets: int
    timestep_targets: int


def _replace_attr_linear(
    parent: nn.Module,
    attr: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    old = getattr(parent, attr)
    if not isinstance(old, nn.Linear):
        raise RuntimeError(
            f"Expected {attr} to be nn.Linear, got {type(old).__name__}"
        )

    setattr(
        parent,
        attr,
        FlowLoRALinear.from_linear(
            old,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ),
    )


def _replace_index_linear(
    parent,
    index: int,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    old = parent[index]
    if not isinstance(old, nn.Linear):
        raise RuntimeError(
            f"Expected indexed module [{index}] to be nn.Linear, "
            f"got {type(old).__name__}"
        )

    parent[index] = FlowLoRALinear.from_linear(
        old,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )


def _inject_all_linears_recursive(
    module: nn.Module,
    *,
    prefix: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    targets: List[str] = []

    for child_name, child in list(module.named_children()):
        path = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear):
            module._modules[child_name] = FlowLoRALinear.from_linear(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            targets.append(path)
        else:
            targets.extend(
                _inject_all_linears_recursive(
                    child,
                    prefix=path,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
            )

    return targets


def inject_flow_conditioning_lora(
    flow: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_attention: bool = True,
    target_ffn: bool = True,
    target_enc_vlm: bool = True,
    target_output: bool = True,
    target_adanorm: bool = False,
    target_timestep: bool = False,
) -> FlowConditioningLoRASummary:
    """
    Inject LoRA into selected parts of the *original* Flow computation path.

    Core target groups:
      - attention: original DiT blocks' attn1 Q/K/V/O
      - ffn: original DiT blocks' FFN input/output projections
      - enc_vlm: Flow VLM-conditioning projection
      - output: DiT.proj_out_1 / proj_out_2

    Optional conditioning groups:
      - AdaNorm linears under block.norm1
      - shared DiT timestep encoder linears

    Important: this function only walks ``DiT.transformer_blocks`` for the
    transformer groups. It intentionally does NOT inject LoRA into any
    ``residual_expert_blocks``. This makes the hybrid experiment clean:
    distributed low-rank correction on the frozen Base path + separate
    nonlinear residual experts.
    """
    dit = getattr(flow, "DiT", None)
    if dit is None:
        raise RuntimeError("policy_backend.flow.DiT does not exist.")

    blocks = getattr(dit, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError(
            "policy_backend.flow.DiT.transformer_blocks does not exist."
        )

    attention_names: List[str] = []
    ffn_names: List[str] = []
    enc_vlm_names: List[str] = []
    output_names: List[str] = []
    adanorm_names: List[str] = []
    timestep_names: List[str] = []

    for idx, block in enumerate(blocks):
        if target_attention:
            attn = getattr(block, "attn1", None)
            if attn is None:
                raise RuntimeError(f"Flow block {idx} has no attn1.")

            for attr in ("to_q", "to_k", "to_v"):
                _replace_attr_linear(
                    attn,
                    attr,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                attention_names.append(
                    f"DiT.transformer_blocks.{idx}.attn1.{attr}"
                )

            to_out = getattr(attn, "to_out", None)
            if to_out is None or len(to_out) < 1:
                raise RuntimeError(
                    f"Flow block {idx} has invalid attn1.to_out."
                )

            _replace_index_linear(
                to_out,
                0,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            attention_names.append(
                f"DiT.transformer_blocks.{idx}.attn1.to_out.0"
            )

        if target_ffn:
            ff = getattr(block, "ff", None)
            if ff is None:
                raise RuntimeError(f"Flow block {idx} has no ff.")

            net = getattr(ff, "net", None)
            if net is None or len(net) < 3:
                raise RuntimeError(f"Flow block {idx} has invalid ff.net.")

            first = net[0]
            if not hasattr(first, "proj"):
                raise RuntimeError(
                    f"Flow block {idx} ff.net[0] has no proj."
                )

            _replace_attr_linear(
                first,
                "proj",
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            ffn_names.append(
                f"DiT.transformer_blocks.{idx}.ff.net.0.proj"
            )

            _replace_index_linear(
                net,
                2,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            ffn_names.append(
                f"DiT.transformer_blocks.{idx}.ff.net.2"
            )

    if target_enc_vlm:
        enc_vlm = getattr(flow, "enc_vlm", None)
        if enc_vlm is None:
            raise RuntimeError("flow.enc_vlm does not exist.")

        if isinstance(enc_vlm, nn.Linear):
            flow.enc_vlm = FlowLoRALinear.from_linear(
                enc_vlm,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            enc_vlm_names = ["enc_vlm"]
        else:
            enc_vlm_names = _inject_all_linears_recursive(
                enc_vlm,
                prefix="enc_vlm",
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

        if len(enc_vlm_names) != 1:
            raise RuntimeError(
                f"Expected exactly 1 enc_vlm Linear, got {len(enc_vlm_names)}"
            )

    if target_output:
        for attr in ("proj_out_1", "proj_out_2"):
            if not hasattr(dit, attr):
                raise RuntimeError(f"flow.DiT.{attr} does not exist.")

            _replace_attr_linear(
                dit,
                attr,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            output_names.append(f"DiT.{attr}")

    if target_adanorm:
        for idx, block in enumerate(blocks):
            norm1 = getattr(block, "norm1", None)
            if norm1 is None:
                raise RuntimeError(
                    f"Requested AdaNorm LoRA but block {idx}.norm1 is missing."
                )

            found = _inject_all_linears_recursive(
                norm1,
                prefix=f"DiT.transformer_blocks.{idx}.norm1",
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            if len(found) != 1:
                raise RuntimeError(
                    f"Expected exactly one nn.Linear under block {idx}.norm1 "
                    f"(AdaLayerNorm.linear), found {len(found)}: {found}"
                )
            adanorm_names.extend(found)

    if target_timestep:
        timestep_encoder = getattr(dit, "timestep_encoder", None)
        if timestep_encoder is None:
            raise RuntimeError(
                "Requested timestep LoRA but flow.DiT.timestep_encoder "
                "does not exist."
            )

        timestep_names = _inject_all_linears_recursive(
            timestep_encoder,
            prefix="DiT.timestep_encoder",
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        if len(timestep_names) != 2:
            raise RuntimeError(
                "Expected exactly 2 nn.Linear modules under "
                "flow.DiT.timestep_encoder, "
                f"found {len(timestep_names)}: {timestep_names}"
            )

    targets = (
        attention_names
        + ffn_names
        + enc_vlm_names
        + output_names
        + adanorm_names
        + timestep_names
    )

    n_blocks = len(blocks)
    expected_attention = 4 * n_blocks if target_attention else 0
    expected_ffn = 2 * n_blocks if target_ffn else 0
    expected_enc_vlm = 1 if target_enc_vlm else 0
    expected_output = 2 if target_output else 0
    expected_adanorm = n_blocks if target_adanorm else 0
    expected_timestep = 2 if target_timestep else 0
    expected = (
        expected_attention
        + expected_ffn
        + expected_enc_vlm
        + expected_output
        + expected_adanorm
        + expected_timestep
    )

    if len(attention_names) != expected_attention:
        raise RuntimeError(
            f"Expected {expected_attention} attention targets, got {len(attention_names)}"
        )
    if len(ffn_names) != expected_ffn:
        raise RuntimeError(
            f"Expected {expected_ffn} FFN targets, got {len(ffn_names)}"
        )
    if len(enc_vlm_names) != expected_enc_vlm:
        raise RuntimeError(
            f"Expected {expected_enc_vlm} enc_vlm targets, got {len(enc_vlm_names)}"
        )
    if len(output_names) != expected_output:
        raise RuntimeError(
            f"Expected {expected_output} output targets, got {len(output_names)}"
        )
    if len(adanorm_names) != expected_adanorm:
        raise RuntimeError(
            f"Expected {expected_adanorm} AdaNorm targets, got {len(adanorm_names)}"
        )
    if len(timestep_names) != expected_timestep:
        raise RuntimeError(
            f"Expected {expected_timestep} timestep targets, got {len(timestep_names)}"
        )
    if len(targets) != expected:
        raise RuntimeError(
            f"Total target mismatch: expected {expected}, got {len(targets)}"
        )
    if not targets:
        raise RuntimeError("No Flow-LoRA targets were selected.")
    if len(targets) != len(set(targets)):
        raise RuntimeError("Duplicate LoRA target names detected.")

    lora_params = [
        (name, p)
        for name, p in flow.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    ]
    if len(lora_params) != 2 * len(targets):
        raise RuntimeError(
            "LoRA tensor count mismatch: "
            f"expected={2 * len(targets)}, got={len(lora_params)}"
        )

    return FlowConditioningLoRASummary(
        num_blocks=n_blocks,
        target_modules=tuple(targets),
        trainable_tensors=len(lora_params),
        trainable_params=sum(p.numel() for _, p in lora_params),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        target_adanorm=bool(target_adanorm),
        target_timestep=bool(target_timestep),
        attention_targets=len(attention_names),
        ffn_targets=len(ffn_names),
        enc_vlm_targets=len(enc_vlm_names),
        output_targets=len(output_names),
        adanorm_targets=len(adanorm_names),
        timestep_targets=len(timestep_names),
    )
