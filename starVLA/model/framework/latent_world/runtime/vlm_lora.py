from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch.nn as nn

from starVLA.model.framework.latent_world.runtime.flow_lora import FlowLoRALinear


TEXT_LINEAR_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
VISION_LINEAR_PATHS = (
    "attn.qkv",
    "attn.proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)
MERGER_LINEAR_PATHS = (
    "linear_fc1",
    "linear_fc2",
)


def _get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _resolve_language_model(vlm):
    for path in ("model.language_model", "language_model"):
        obj = _get_nested_attr(vlm, path)
        if obj is not None:
            return obj, path
    raise RuntimeError("Failed to resolve VLM language model (expected model.language_model or language_model).")


def _resolve_text_layers(vlm):
    lm, lm_path = _resolve_language_model(vlm)
    layers = getattr(lm, "layers", None)
    if not isinstance(layers, (nn.ModuleList, list)):
        raise RuntimeError(f"Failed to resolve text layers at {lm_path}.layers")
    return layers, f"{lm_path}.layers"


def _resolve_visual(vlm):
    for path in ("model.visual", "visual"):
        obj = _get_nested_attr(vlm, path)
        if obj is not None:
            return obj, path
    raise RuntimeError("Failed to resolve VLM visual module (expected model.visual or visual).")


def _resolve_parent_and_attr(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential, list)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _replace_linear_at_path(
    root: nn.Module,
    path: str,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    parent, attr = _resolve_parent_and_attr(root, path)
    if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential, list)):
        old = parent[int(attr)]
        if not isinstance(old, nn.Linear):
            raise RuntimeError(f"Expected nn.Linear at {path}, got {type(old).__name__}")
        parent[int(attr)] = FlowLoRALinear.from_linear(old, rank=rank, alpha=alpha, dropout=dropout)
        return
    old = getattr(parent, attr)
    if not isinstance(old, nn.Linear):
        raise RuntimeError(f"Expected nn.Linear at {path}, got {type(old).__name__}")
    setattr(parent, attr, FlowLoRALinear.from_linear(old, rank=rank, alpha=alpha, dropout=dropout))


@dataclass(frozen=True)
class VLMLoRASummary:
    target_modules: Tuple[str, ...]
    trainable_tensors: int
    trainable_params: int
    rank: int
    alpha: float
    dropout: float
    text_targets: int
    vision_targets: int
    merger_targets: int
    text_layers_total: int
    text_layers_selected: Tuple[int, ...]
    vision_blocks_total: int
    merger_modules_total: int


def inject_vlm_lora(
    vlm: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_text: bool = False,
    text_last_n: int = 0,
    target_vision: bool = False,
    target_merger: bool = False,
) -> VLMLoRASummary:
    """Inject task LoRA into selected Qwen3-VL substructures.

    Text target:
      q/k/v/o + gate/up/down in the retained decoder layers.
      ``text_last_n <= 0`` means all retained layers (LaWAM currently keeps 16).

    Vision target:
      qkv/proj + MLP fc1/fc2 in every visual transformer block.
      Patch embedding, position embedding and mergers are intentionally excluded.

    Merger target:
      linear_fc1/fc2 in the main merger and each DeepStack merger.
    """
    if rank <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")
    if not (target_text or target_vision or target_merger):
        raise RuntimeError("VLM-LoRA requires at least one VLM target group.")

    targets: List[str] = []
    text_names: List[str] = []
    vision_names: List[str] = []
    merger_names: List[str] = []
    selected_text_indices: Tuple[int, ...] = tuple()
    text_layers_total = 0
    vision_blocks_total = 0
    merger_modules_total = 0

    if target_text:
        layers, layers_prefix = _resolve_text_layers(vlm)
        text_layers_total = len(layers)
        if text_layers_total <= 0:
            raise RuntimeError("VLM text layers are empty.")
        if int(text_last_n) > 0:
            n = min(int(text_last_n), text_layers_total)
            selected_text_indices = tuple(range(text_layers_total - n, text_layers_total))
        else:
            selected_text_indices = tuple(range(text_layers_total))

        for idx in selected_text_indices:
            layer = layers[idx]
            for local_path in TEXT_LINEAR_PATHS:
                _replace_linear_at_path(
                    layer,
                    local_path,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                text_names.append(f"{layers_prefix}.{idx}.{local_path}")

    if target_vision or target_merger:
        visual, visual_prefix = _resolve_visual(vlm)

        if target_vision:
            blocks = getattr(visual, "blocks", None)
            if not isinstance(blocks, (nn.ModuleList, list)):
                raise RuntimeError(f"Failed to resolve visual blocks at {visual_prefix}.blocks")
            vision_blocks_total = len(blocks)
            if vision_blocks_total <= 0:
                raise RuntimeError("VLM visual blocks are empty.")
            for idx, block in enumerate(blocks):
                for local_path in VISION_LINEAR_PATHS:
                    _replace_linear_at_path(
                        block,
                        local_path,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    vision_names.append(f"{visual_prefix}.blocks.{idx}.{local_path}")

        if target_merger:
            merger_modules = []
            main_merger = getattr(visual, "merger", None)
            if main_merger is None:
                raise RuntimeError(f"{visual_prefix}.merger does not exist")
            merger_modules.append((f"{visual_prefix}.merger", main_merger))

            deepstack = getattr(visual, "deepstack_merger_list", None)
            if deepstack is not None:
                for idx, module in enumerate(deepstack):
                    merger_modules.append((f"{visual_prefix}.deepstack_merger_list.{idx}", module))
            merger_modules_total = len(merger_modules)

            for prefix, merger in merger_modules:
                for local_path in MERGER_LINEAR_PATHS:
                    _replace_linear_at_path(
                        merger,
                        local_path,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    merger_names.append(f"{prefix}.{local_path}")

    targets = text_names + vision_names + merger_names
    if len(targets) != len(set(targets)):
        raise RuntimeError("Duplicate VLM-LoRA target names detected.")

    lora_params = [
        (name, p)
        for name, p in vlm.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    ]
    if len(lora_params) != 2 * len(targets):
        raise RuntimeError(
            f"VLM-LoRA tensor mismatch: targets={len(targets)}, tensors={len(lora_params)}"
        )

    # Architecture sanity checks for the current LaWAM Qwen3-VL setup.
    if target_text and len(text_names) != 7 * len(selected_text_indices):
        raise RuntimeError(
            f"Unexpected text target count: {len(text_names)} for layers={selected_text_indices}"
        )
    if target_vision and len(vision_names) != 4 * vision_blocks_total:
        raise RuntimeError(
            f"Unexpected vision target count: {len(vision_names)} for {vision_blocks_total} blocks"
        )
    if target_merger and len(merger_names) != 2 * merger_modules_total:
        raise RuntimeError(
            f"Unexpected merger target count: {len(merger_names)} for {merger_modules_total} mergers"
        )

    return VLMLoRASummary(
        target_modules=tuple(targets),
        trainable_tensors=len(lora_params),
        trainable_params=sum(p.numel() for _, p in lora_params),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
        text_targets=len(text_names),
        vision_targets=len(vision_names),
        merger_targets=len(merger_names),
        text_layers_total=text_layers_total,
        text_layers_selected=selected_text_indices,
        vision_blocks_total=vision_blocks_total,
        merger_modules_total=merger_modules_total,
    )
