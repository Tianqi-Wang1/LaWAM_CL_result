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
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


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


def canonical_flow_map(state: dict, *, include_residual: bool = True) -> dict[str, torch.Tensor]:
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
        ck = CANONICAL + suffix
        ak = ALIAS + suffix
        if ck in state and ak in state and torch.is_tensor(state[ck]) and torch.is_tensor(state[ak]):
            if not tensor_equal(state[ck], state[ak]):
                raise RuntimeError(f"Canonical/alias mismatch inside checkpoint: {ck} vs {ak}")
        if ck in state:
            out[suffix] = state[ck]
        elif ak in state:
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
            continue
        if not tensor_equal(value, final[key]):
            changed.append(key)
    if missing:
        raise RuntimeError(f"Final checkpoint is missing Base upstream keys: {missing[:30]}")
    if changed:
        raise RuntimeError(
            "Frozen upstream changed unexpectedly: "
            f"count={len(changed)} examples={changed[:30]}"
        )
    return checked


def verify_residual(args):
    base = load_state(args.base_ckpt)
    source = load_state(args.flow_source_ckpt)
    final = load_state(args.final_ckpt)

    upstream_checked = verify_upstream_exact(base, final)
    base_core = canonical_flow_map(base, include_residual=False)
    source_core = canonical_flow_map(source, include_residual=False)
    final_core = canonical_flow_map(final, include_residual=False)
    final_all = canonical_flow_map(final, include_residual=True)

    if set(source_core) != set(final_core):
        missing = sorted(set(source_core) - set(final_core))
        extra = sorted(set(final_core) - set(source_core))
        raise RuntimeError(
            f"Flow-core key mismatch source vs final: missing={missing[:20]} extra={extra[:20]}"
        )

    core_changed = [suffix for suffix in source_core if not tensor_equal(source_core[suffix], final_core[suffix])]
    if core_changed:
        raise RuntimeError(
            "Frozen Flow core changed during residual training: "
            f"count={len(core_changed)} examples={core_changed[:30]}"
        )

    source_vs_base = []
    common = sorted(set(base_core).intersection(source_core))
    for suffix in common:
        if not tensor_equal(base_core[suffix], source_core[suffix]):
            source_vs_base.append(suffix)

    residual = {k: v for k, v in final_all.items() if k.startswith(RESIDUAL)}
    if not residual:
        raise RuntimeError("Final checkpoint contains no residual expert tensors.")

    block_ids = sorted({int(k.split(".")[2]) for k in residual if len(k.split(".")) > 2 and k.split(".")[2].isdigit()})
    expected = list(range(args.num_residual_blocks))
    if block_ids != expected:
        raise RuntimeError(f"Unexpected residual block ids: got={block_ids}, expected={expected}")

    out_proj = {k: v for k, v in residual.items() if k.endswith("out_proj.weight")}
    if len(out_proj) != args.num_residual_blocks:
        raise RuntimeError(
            f"Expected {args.num_residual_blocks} residual out_proj weights, found {len(out_proj)}"
        )
    nonzero_out_proj = [k for k, v in out_proj.items() if int(torch.count_nonzero(v).item()) > 0]
    if not nonzero_out_proj:
        raise RuntimeError(
            "All residual out_proj weights are still exactly zero; optimizer/training likely did not update the expert."
        )

    residual_params = sum(v.numel() for v in residual.values() if torch.is_tensor(v))
    core_params = sum(v.numel() for v in final_core.values() if torch.is_tensor(v))
    print(
        "[OK] Residual-Expert checkpoint verified\n"
        f"  upstream_exact_tensors : {upstream_checked}\n"
        f"  frozen_flow_core       : {len(final_core)} tensors / {core_params:,} params\n"
        f"  residual_expert        : {len(residual)} tensors / {residual_params:,} params\n"
        f"  residual/core params   : {residual_params / max(core_params,1):.4f}\n"
        f"  out_proj_updated       : {len(nonzero_out_proj)}/{len(out_proj)}\n"
        f"  source-vs-base changed : {len(source_vs_base)}/{len(common)} flow tensors"
    )


def verify_full_flow(args):
    base = load_state(args.base_ckpt)
    final = load_state(args.final_ckpt)
    upstream_checked = verify_upstream_exact(base, final)
    base_core = canonical_flow_map(base, include_residual=False)
    final_core = canonical_flow_map(final, include_residual=False)
    if set(base_core) != set(final_core):
        raise RuntimeError("Full-flow control has a core-key mismatch vs Base.")
    changed = [suffix for suffix in base_core if not tensor_equal(base_core[suffix], final_core[suffix])]
    if not changed:
        raise RuntimeError("No Flow-core tensor changed in full-flow control.")
    residual = [k for k in canonical_flow_map(final, include_residual=True) if k.startswith(RESIDUAL)]
    if residual:
        raise RuntimeError(f"Full-flow control unexpectedly contains residual-expert tensors: {residual[:10]}")
    print(
        "[OK] Full-Flow control checkpoint verified\n"
        f"  upstream_exact_tensors : {upstream_checked}\n"
        f"  flow_changed_tensors   : {len(changed)}/{len(base_core)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["residual", "full_flow"], required=True)
    ap.add_argument("--base-ckpt", type=Path, required=True)
    ap.add_argument("--final-ckpt", type=Path, required=True)
    ap.add_argument("--flow-source-ckpt", type=Path)
    ap.add_argument("--num-residual-blocks", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "residual":
        if args.flow_source_ckpt is None:
            ap.error("--flow-source-ckpt is required for residual mode")
        if args.num_residual_blocks <= 0:
            ap.error("--num-residual-blocks must be > 0 for residual mode")
        verify_residual(args)
    else:
        verify_full_flow(args)


if __name__ == "__main__":
    main()
