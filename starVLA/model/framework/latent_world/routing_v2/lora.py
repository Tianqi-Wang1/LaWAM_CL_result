from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch.nn as nn

from starVLA.model.framework.latent_world.runtime.flow_lora import FlowLoRALinear


@dataclass(frozen=True)
class RecursiveLoRASummary:
    target_modules: Tuple[str, ...]
    skipped_modules: Tuple[str, ...]
    trainable_tensors: int
    trainable_params: int
    rank: int
    alpha: float
    dropout: float


def _collect_linear_paths(
    module: nn.Module,
    prefix: str = "",
) -> tuple[list[str], list[str]]:
    """Collect *functionally invoked* ``nn.Linear`` modules for generic LoRA.

    A subtle PyTorch detail matters here: ``nn.MultiheadAttention.forward`` does
    **not** call ``self.out_proj(x)``.  Instead it passes ``out_proj.weight`` and
    ``out_proj.bias`` directly to ``torch.nn.functional.multi_head_attention_forward``.
    Therefore replacing ``MultiheadAttention.out_proj`` by a LoRA ``Linear`` creates
    trainable A/B parameters that never participate in autograd.  Under DDP this
    triggers ``Expected to have finished reduction ... parameters were not used``.

    V2 intentionally keeps packed/raw MHA Q/K/V parameters frozen, so we also skip
    the functionally-bypassed ``out_proj`` here.  FFNs, AdaLN projections and other
    explicit Linear modules are still adapted normally.
    """
    targets: list[str] = []
    skipped: list[str] = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name

        # PyTorch MHA consumes out_proj.weight/bias functionally; the module's
        # ``forward`` (and hence a LoRA residual placed there) is never invoked.
        if isinstance(module, nn.MultiheadAttention) and name == "out_proj":
            skipped.append(path)
            continue

        if isinstance(child, FlowLoRALinear):
            targets.append(path)
            continue
        if isinstance(child, nn.Linear):
            targets.append(path)
            continue

        child_targets, child_skipped = _collect_linear_paths(child, path)
        targets.extend(child_targets)
        skipped.extend(child_skipped)
    return targets, skipped


def _resolve_parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_recursive_linear_lora(
    module: nn.Module,
    *,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
    module_name: str = "module",
) -> RecursiveLoRASummary:
    """Replace every explicit nn.Linear below ``module`` with Base+LoRA.

    Notes
    -----
    ``nn.MultiheadAttention`` stores Q/K/V projections as raw parameters rather than
    child ``nn.Linear`` modules.  This intentionally keeps V2 simple: LoRA is added
    to all *explicit* linear projections (FFNs, output projections, AdaLN modulation,
    input/output projections, etc.) while packed attention projection tensors remain
    frozen Base parameters.  The runtime summary prints the exact target list.
    """
    if int(rank) <= 0:
        raise ValueError(f"{module_name}: LoRA rank must be > 0, got {rank}")

    paths, skipped_paths = _collect_linear_paths(module)
    # If a module was already injected, _iter_linear_paths reports it.  Only replace
    # plain nn.Linear targets; existing FlowLoRALinear modules are validated by
    # from_linear and left structurally unchanged.
    for path in paths:
        parent, attr = _resolve_parent(module, path)
        if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            old = parent[int(attr)]
            parent[int(attr)] = FlowLoRALinear.from_linear(
                old, rank=int(rank), alpha=float(alpha), dropout=float(dropout)
            )
        else:
            old = getattr(parent, attr)
            setattr(
                parent,
                attr,
                FlowLoRALinear.from_linear(
                    old, rank=int(rank), alpha=float(alpha), dropout=float(dropout)
                ),
            )

    lora_params = [
        (name, p)
        for name, p in module.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    ]
    expected_tensors = 2 * len(paths)
    if len(lora_params) != expected_tensors:
        raise RuntimeError(
            f"{module_name}: LoRA tensor mismatch: targets={len(paths)}, "
            f"expected_tensors={expected_tensors}, got={len(lora_params)}"
        )

    return RecursiveLoRASummary(
        target_modules=tuple(paths),
        skipped_modules=tuple(skipped_paths),
        trainable_tensors=len(lora_params),
        trainable_params=sum(p.numel() for _, p in lora_params),
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
    )


def set_lora_trainable(module: nn.Module, trainable: bool) -> int:
    count = 0
    for name, p in module.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            p.requires_grad_(bool(trainable))
            count += p.numel()
    return count


@contextmanager
def temporarily_disable_lora(module: nn.Module):
    """Temporarily set LoRA residual scaling to zero while preserving Base weights."""
    touched = []
    for child in module.modules():
        if isinstance(child, FlowLoRALinear):
            touched.append((child, float(child.lora_scaling)))
            child.lora_scaling = 0.0
    try:
        yield
    finally:
        for child, scaling in touched:
            child.lora_scaling = scaling
