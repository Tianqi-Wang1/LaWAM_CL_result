#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

CANONICAL = "policy_backend.flow."
ALIAS = "policy_action_head."
RESIDUAL = "DiT.residual_expert_blocks."


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


def is_lora_suffix(suffix: str) -> bool:
    return suffix.endswith(".lora_A") or suffix.endswith(".lora_B")


def canonical_flow_map(
    state: dict,
    *,
    include_residual: bool = True,
    include_lora: bool = True,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    suffixes = set()
    for key in state:
        if key.startswith(CANONICAL):
            suffixes.add(key[len(CANONICAL):])
        elif key.startswith(ALIAS):
            suffixes.add(key[len(ALIAS):])

    for suffix in sorted(suffixes):
        if (not include_residual) and suffix.startswith(RESIDUAL):
            continue
        if (not include_lora) and is_lora_suffix(suffix):
            continue
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
    changed = []
    missing = []
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


def verify_residual_tensors(final_all: dict, expected_blocks: int) -> tuple[int, int, int]:
    residual = {k: v for k, v in final_all.items() if k.startswith(RESIDUAL)}
    if not residual:
        raise RuntimeError("Final checkpoint contains no residual expert tensors.")

    block_ids = sorted({
        int(k.split(".")[2])
        for k in residual
        if len(k.split(".")) > 2 and k.split(".")[2].isdigit()
    })
    expected_ids = list(range(expected_blocks))
    if block_ids != expected_ids:
        raise RuntimeError(
            f"Unexpected residual ModuleList ids: got={block_ids}, expected={expected_ids}"
        )

    out_proj = {k: v for k, v in residual.items() if k.endswith("out_proj.weight")}
    if len(out_proj) != expected_blocks:
        raise RuntimeError(
            f"Expected {expected_blocks} residual out_proj weights, found {len(out_proj)}"
        )
    updated = [k for k, v in out_proj.items() if int(torch.count_nonzero(v).item()) > 0]
    if not updated:
        raise RuntimeError("All residual out_proj weights are still exactly zero.")

    params = sum(v.numel() for v in residual.values())
    return len(residual), params, len(updated)


def verify_residual(args):
    base = load_state(args.base_ckpt)
    final = load_state(args.final_ckpt)
    upstream_checked = verify_upstream_exact(base, final)

    base_core = canonical_flow_map(base, include_residual=False, include_lora=False)
    final_core = canonical_flow_map(final, include_residual=False, include_lora=False)
    if set(base_core) != set(final_core):
        missing = sorted(set(base_core) - set(final_core))
        extra = sorted(set(final_core) - set(base_core))
        raise RuntimeError(
            f"Frozen Flow-core key mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )
    changed = [k for k in base_core if not tensor_equal(base_core[k], final_core[k])]
    if changed:
        raise RuntimeError(
            f"Frozen Base Flow core changed: count={len(changed)} examples={changed[:30]}"
        )

    final_all = canonical_flow_map(final, include_residual=True, include_lora=True)
    n_tensors, n_params, updated = verify_residual_tensors(
        final_all, args.num_residual_blocks
    )
    lora = [k for k in final_all if is_lora_suffix(k)]
    if lora:
        raise RuntimeError(f"Residual-only run unexpectedly contains LoRA tensors: {lora[:20]}")

    core_params = sum(v.numel() for v in final_core.values())
    print(
        "[OK] Residual-only checkpoint verified\n"
        f"  upstream_exact_tensors : {upstream_checked}\n"
        f"  frozen_flow_core       : {len(final_core)} tensors / {core_params:,} params\n"
        f"  residual_expert        : {n_tensors} tensors / {n_params:,} params\n"
        f"  residual/core params   : {n_params / max(core_params, 1):.4f}\n"
        f"  out_proj_updated       : {updated}/{args.num_residual_blocks}"
    )


def verify_hybrid(args):
    base = load_state(args.base_ckpt)
    final = load_state(args.final_ckpt)
    upstream_checked = verify_upstream_exact(base, final)

    base_core = canonical_flow_map(base, include_residual=False, include_lora=False)
    final_core = canonical_flow_map(final, include_residual=False, include_lora=False)
    if set(base_core) != set(final_core):
        missing = sorted(set(base_core) - set(final_core))
        extra = sorted(set(final_core) - set(base_core))
        raise RuntimeError(
            f"Hybrid frozen Flow-core key mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )
    changed = [k for k in base_core if not tensor_equal(base_core[k], final_core[k])]
    if changed:
        raise RuntimeError(
            "Original Base Flow weights changed before merge: "
            f"count={len(changed)} examples={changed[:30]}"
        )

    final_all = canonical_flow_map(final, include_residual=True, include_lora=True)
    n_tensors, n_params, updated = verify_residual_tensors(
        final_all, args.num_residual_blocks
    )
    lora = {k: v for k, v in final_all.items() if is_lora_suffix(k)}
    if not lora:
        raise RuntimeError("Hybrid checkpoint contains no LoRA tensors.")
    a_keys = sorted(k for k in lora if k.endswith(".lora_A"))
    b_keys = sorted(k for k in lora if k.endswith(".lora_B"))
    if len(a_keys) != args.expected_lora_targets or len(b_keys) != args.expected_lora_targets:
        raise RuntimeError(
            f"Expected {args.expected_lora_targets} LoRA targets, "
            f"got A={len(a_keys)}, B={len(b_keys)}"
        )
    nonzero_b = sum(int(torch.count_nonzero(lora[k]).item()) > 0 for k in b_keys)
    if nonzero_b == 0:
        raise RuntimeError("All LoRA B tensors are still exactly zero.")
    lora_params = sum(v.numel() for v in lora.values())

    print(
        "[OK] Residual+LoRA checkpoint verified\n"
        f"  upstream_exact_tensors : {upstream_checked}\n"
        f"  frozen_flow_core       : {len(final_core)} tensors\n"
        f"  residual_expert        : {n_tensors} tensors / {n_params:,} params\n"
        f"  residual_out_updated   : {updated}/{args.num_residual_blocks}\n"
        f"  LoRA targets           : {len(a_keys)}\n"
        f"  LoRA params            : {lora_params:,}\n"
        f"  LoRA B updated         : {nonzero_b}/{len(b_keys)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["residual", "hybrid"], required=True)
    ap.add_argument("--base-ckpt", type=Path, required=True)
    ap.add_argument("--final-ckpt", type=Path, required=True)
    ap.add_argument("--num-residual-blocks", type=int, required=True)
    ap.add_argument("--expected-lora-targets", type=int, default=96)
    args = ap.parse_args()

    if args.num_residual_blocks <= 0:
        ap.error("--num-residual-blocks must be > 0")
    if args.mode == "residual":
        verify_residual(args)
    else:
        verify_hybrid(args)


if __name__ == "__main__":
    main()
