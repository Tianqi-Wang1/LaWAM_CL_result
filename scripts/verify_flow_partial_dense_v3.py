#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

CANONICAL = "policy_backend.flow."
ALIAS = "policy_action_head."


def load_state(path: Path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)
            if isinstance(nested, dict) and nested and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    return obj


def tensor_equal(a, b) -> bool:
    return (
        torch.is_tensor(a)
        and torch.is_tensor(b)
        and tuple(a.shape) == tuple(b.shape)
        and a.dtype == b.dtype
        and torch.equal(a, b)
    )


def is_flow_key(key: str) -> bool:
    return key.startswith(CANONICAL) or key.startswith(ALIAS)


def canonical_flow_map(state: dict) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    suffixes = set()
    for key in state:
        if key.startswith(CANONICAL):
            suffixes.add(key[len(CANONICAL):])
        elif key.startswith(ALIAS):
            suffixes.add(key[len(ALIAS):])

    for suffix in sorted(suffixes):
        ck = CANONICAL + suffix
        ak = ALIAS + suffix
        if ck in state and ak in state and torch.is_tensor(state[ck]) and torch.is_tensor(state[ak]):
            if not tensor_equal(state[ck], state[ak]):
                raise RuntimeError(f"Canonical/alias mismatch: {ck} vs {ak}")
        if ck in state and torch.is_tensor(state[ck]):
            out[suffix] = state[ck]
        elif ak in state and torch.is_tensor(state[ak]):
            out[suffix] = state[ak]
    return out


def verify_upstream_exact(base: dict, final: dict) -> int:
    checked = 0
    missing = []
    changed = []
    for key, value in base.items():
        if not torch.is_tensor(value) or is_flow_key(key):
            continue
        checked += 1
        if key not in final:
            missing.append(key)
        elif not tensor_equal(value, final[key]):
            changed.append(key)
    if missing:
        raise RuntimeError(f"Missing frozen upstream keys: {missing[:30]}")
    if changed:
        raise RuntimeError(
            f"Frozen upstream changed: count={len(changed)} examples={changed[:30]}"
        )
    return checked


def parse_layers(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("No layers provided")
    if len(vals) != len(set(vals)):
        raise ValueError(f"Duplicate layers: {vals}")
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=Path, required=True)
    ap.add_argument("--final-ckpt", type=Path, required=True)
    ap.add_argument("--layers", type=str, required=True, help="Comma separated DiT block indices")
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    allowed_prefixes = tuple(f"DiT.transformer_blocks.{idx}." for idx in layers)

    base = load_state(args.base_ckpt)
    final = load_state(args.final_ckpt)
    upstream_checked = verify_upstream_exact(base, final)

    base_flow = canonical_flow_map(base)
    final_flow = canonical_flow_map(final)
    if set(base_flow) != set(final_flow):
        missing = sorted(set(base_flow) - set(final_flow))
        extra = sorted(set(final_flow) - set(base_flow))
        raise RuntimeError(
            f"Flow key mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )

    residual = [k for k in final_flow if "residual_expert_blocks." in k]
    lora = [k for k in final_flow if k.endswith(".lora_A") or k.endswith(".lora_B")]
    if residual:
        raise RuntimeError(f"Partial-dense run unexpectedly contains residual expert tensors: {residual[:20]}")
    if lora:
        raise RuntimeError(f"Partial-dense run unexpectedly contains LoRA tensors: {lora[:20]}")

    changed = []
    changed_outside = []
    for key, value in base_flow.items():
        if not tensor_equal(value, final_flow[key]):
            changed.append(key)
            if not key.startswith(allowed_prefixes):
                changed_outside.append(key)
    if changed_outside:
        raise RuntimeError(
            "Non-selected Flow parameters changed: "
            f"count={len(changed_outside)} examples={changed_outside[:30]}"
        )

    selected_keys = [k for k in base_flow if k.startswith(allowed_prefixes)]
    if not selected_keys:
        raise RuntimeError(f"No selected Flow tensors found for layers={layers}")
    selected_params = sum(base_flow[k].numel() for k in selected_keys)

    per_layer = {}
    total_changed_params = 0
    for idx in layers:
        prefix = f"DiT.transformer_blocks.{idx}."
        keys = [k for k in selected_keys if k.startswith(prefix)]
        changed_keys = [k for k in keys if not tensor_equal(base_flow[k], final_flow[k])]
        if not changed_keys:
            raise RuntimeError(
                f"Selected dense layer {idx} has no changed tensors; training likely failed."
            )
        n_params = sum(base_flow[k].numel() for k in keys)
        changed_params = sum(base_flow[k].numel() for k in changed_keys)
        total_changed_params += changed_params
        per_layer[idx] = {
            "tensors": len(keys),
            "params": n_params,
            "changed_tensors": len(changed_keys),
            "changed_params": changed_params,
        }

    frozen_flow_tensors = len(base_flow) - len(selected_keys)
    frozen_flow_params = sum(
        v.numel() for k, v in base_flow.items() if not k.startswith(allowed_prefixes)
    )

    print(
        "[OK] Partial-dense Flow checkpoint verified\n"
        f"  layers                 : {layers}\n"
        f"  upstream_exact_tensors : {upstream_checked}\n"
        f"  selected_flow          : {len(selected_keys)} tensors / {selected_params:,} params\n"
        f"  changed_selected       : {len(changed)} tensors / {total_changed_params:,} params\n"
        f"  frozen_flow            : {frozen_flow_tensors} tensors / {frozen_flow_params:,} params\n"
        f"  per_layer              : {per_layer}\n"
        "  residual_expert        : none\n"
        "  LoRA                   : none"
    )


if __name__ == "__main__":
    main()
