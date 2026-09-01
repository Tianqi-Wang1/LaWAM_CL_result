#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import torch


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
        for key in ("state_dict", "model"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and all(torch.is_tensor(v) for v in inner.values()):
                return dict(inner)
    raise RuntimeError(f"Unsupported checkpoint structure: {path}")


def is_lora(k: str) -> bool:
    return k.endswith(".lora_A") or k.endswith(".lora_B")


def _normalize_alias_prefix(prefix: str) -> str:
    """Normalize state-dict aliases to the canonical policy_backend.* namespace.

    LaWAM registers the same VLM under both policy_backend.vlm and
    policy_vlm_adapter.model.  PyTorch therefore serializes LoRA tensors under
    both paths, exactly like the Flow alias policy_action_head.  These are
    aliases of the same module, not unintended LoRA injection.
    """
    if prefix.startswith("policy_vlm_adapter.model."):
        return "policy_backend.vlm." + prefix[len("policy_vlm_adapter.model."): ]
    return prefix


def classify_prefix(prefix: str) -> str:
    prefix = _normalize_alias_prefix(prefix)

    # Flow: ONLY original DiT transformer blocks.
    if re.match(r"^(policy_backend\.flow|policy_action_head)\.DiT\.transformer_blocks\.\d+\.", prefix):
        return "flow_dit"

    # Text.  Depending on the wrapper, the HF model may contribute one extra
    # `model.` level, so accept both canonical forms after alias normalization.
    if (
        prefix.startswith("policy_backend.vlm.model.language_model.layers.")
        or prefix.startswith("policy_backend.vlm.language_model.layers.")
        or prefix.startswith("policy_backend.vlm.model.model.language_model.layers.")
    ):
        return "vlm_text"

    # Vision transformer blocks.
    if re.match(r"^policy_backend\.vlm\.(model\.){0,2}visual\.blocks\.\d+\.", prefix):
        return "vlm_vision"

    # Main + deepstack mergers.
    if re.match(r"^policy_backend\.vlm\.(model\.){0,2}visual\.merger\.", prefix):
        return "vlm_merger"
    if re.match(r"^policy_backend\.vlm\.(model\.){0,2}visual\.deepstack_merger_list\.", prefix):
        return "vlm_merger"
    return "other"


def expected_variant_groups(variant: str):
    table = {
        "text_last4": {"vlm_text"},
        "text_last8": {"vlm_text"},
        "text_all": {"vlm_text"},
        "vision": {"vlm_vision"},
        "merger": {"vlm_merger"},
        "text_vision": {"vlm_text", "vlm_vision"},
        "full": {"vlm_text", "vlm_vision", "vlm_merger"},
    }
    if variant not in table:
        raise ValueError(f"Unknown variant: {variant}")
    return table[variant]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--unmerged", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    base = load_state(args.base)
    raw = load_state(args.unmerged)
    merged = load_state(args.merged)

    # Unmerged: every original Base tensor must be exactly preserved.
    missing = [k for k in base if k not in raw]
    if missing:
        raise RuntimeError(f"Unmerged checkpoint is missing Base keys: {missing[:20]}")
    changed_base = [k for k in base if not torch.equal(base[k], raw[k])]
    if changed_base:
        raise RuntimeError(f"Base tensors changed before LoRA merge: {changed_base[:20]}")

    extras = sorted(k for k in raw if k not in base)
    if not extras or any(not is_lora(k) for k in extras):
        bad = [k for k in extras if not is_lora(k)]
        raise RuntimeError(f"Unexpected non-LoRA extra keys: {bad[:20]}")
    a_keys = [k for k in extras if k.endswith(".lora_A")]
    b_keys = [k for k in extras if k.endswith(".lora_B")]
    if len(a_keys) != len(b_keys):
        raise RuntimeError(f"A/B count mismatch: {len(a_keys)} vs {len(b_keys)}")

    group_counts = {g: 0 for g in ("flow_dit", "vlm_text", "vlm_vision", "vlm_merger", "other")}
    unique_prefixes = set()
    lora_params = 0
    for k in extras:
        prefix = k.rsplit(".lora_", 1)[0]
        unique_prefixes.add(prefix)
        group_counts[classify_prefix(prefix)] += 1
        lora_params += raw[k].numel()
        if k.endswith(".lora_A") and int(raw[k].shape[0]) != int(args.rank):
            raise RuntimeError(f"Wrong rank for {k}: {tuple(raw[k].shape)}")

    if group_counts["other"]:
        bad = sorted(p for p in unique_prefixes if classify_prefix(p) == "other")
        raise RuntimeError(f"LoRA leaked to unexpected modules: {bad[:20]}")
    if group_counts["flow_dit"] <= 0:
        raise RuntimeError("No Flow DiT LoRA detected")

    expected_vlm = expected_variant_groups(args.variant)
    actual_vlm = {g for g in ("vlm_text", "vlm_vision", "vlm_merger") if group_counts[g] > 0}
    if actual_vlm != expected_vlm:
        raise RuntimeError(f"VLM target mismatch: actual={actual_vlm}, expected={expected_vlm}")

    # Flow LoRA must not touch any Flow interface. Prefix classifier guarantees only transformer_blocks.
    flow_prefixes = [p for p in unique_prefixes if classify_prefix(p) == "flow_dit"]
    if any("enc_vlm" in p or "proj_out" in p or "action_encoder" in p or "action_decoder" in p for p in flow_prefixes):
        raise RuntimeError("Flow LoRA leaked into interface modules")

    # Merged checkpoint: only merged target weights may differ from Base.
    if any(is_lora(k) for k in merged):
        raise RuntimeError("Merged checkpoint still contains LoRA keys")
    extra_merged = [k for k in merged if k not in base]
    missing_merged = [k for k in base if k not in merged]
    if extra_merged or missing_merged:
        raise RuntimeError(
            f"Merged key mismatch: extra={extra_merged[:10]}, missing={missing_merged[:10]}"
        )

    target_weight_keys = {p + ".weight" for p in unique_prefixes}
    non_target_changes = []
    target_changed = {"flow_dit": 0, "vlm_text": 0, "vlm_vision": 0, "vlm_merger": 0}
    for k in base:
        eq = torch.equal(base[k], merged[k])
        if k in target_weight_keys:
            if not eq:
                prefix = k[: -len(".weight")]
                g = classify_prefix(prefix)
                if g in target_changed:
                    target_changed[g] += 1
        elif not eq:
            non_target_changes.append(k)
    if non_target_changes:
        raise RuntimeError(f"Merged checkpoint changed non-target Base tensors: {non_target_changes[:20]}")
    if target_changed["flow_dit"] <= 0:
        raise RuntimeError("No Flow DiT merged weight changed")
    for g in expected_vlm:
        if target_changed[g] <= 0:
            raise RuntimeError(f"No merged weight changed in expected VLM group {g}")

    print("[OK] Joint VLM+Flow LoRA checkpoint audit passed")
    print(f"  variant              : {args.variant}")
    print(f"  LoRA tensors         : {len(extras)}")
    print(f"  unique LoRA modules  : {len(unique_prefixes)}")
    print(f"  LoRA parameters      : {lora_params:,}")
    print(f"  group tensor counts  : {group_counts}")
    print(f"  merged changed       : {target_changed}")
    print("  Base tensors before merge: bitwise exact")
    print("  Non-target tensors after merge: bitwise exact")


if __name__ == "__main__":
    main()
