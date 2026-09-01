#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import torch

TEXT_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


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


def canonical_key(k: str) -> str:
    if k.startswith("policy_action_head."):
        return "policy_backend.flow." + k[len("policy_action_head."):]
    if k.startswith("policy_vlm_adapter.model."):
        return "policy_backend.vlm." + k[len("policy_vlm_adapter.model."):]
    return k


def canonical_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        ck = canonical_key(k)
        if ck in out:
            if not torch.equal(out[ck], v):
                raise RuntimeError(f"Alias mismatch for {ck}")
        else:
            out[ck] = v
    return out


def is_lora(k: str) -> bool:
    return k.endswith(".lora_A") or k.endswith(".lora_B")


def flow_block_idx(k: str):
    m = re.match(r"^policy_backend\.flow\.DiT\.transformer_blocks\.(\d+)\.", k)
    return None if m is None else int(m.group(1))


def nonlinear_block_idx(k: str):
    if ".attn_nonlinear_adapter." not in k and ".ffn_nonlinear_adapter." not in k:
        return None
    return flow_block_idx(k)


def is_text_module_prefix(prefix: str) -> bool:
    if not prefix.startswith("policy_backend.vlm."):
        return False
    if ".language_model.layers." not in prefix:
        return False
    return any(prefix.endswith(s) for s in TEXT_SUFFIXES)


def is_conditioning_adapter(k: str) -> bool:
    return k.startswith("policy_backend.flow.") and ".conditioning_adapter." in k


def is_nonlinear_adapter(k: str) -> bool:
    return k.startswith("policy_backend.flow.") and (
        ".attn_nonlinear_adapter." in k or ".ffn_nonlinear_adapter." in k
    )


def expected(mode: str):
    # Exact canonical budgets for LaWAM hidden_dim=1024, nonlinear bottleneck=128.
    # Nonlinear adapter = 1024*128 + 128*1024 = 262,144 params per branch,
    # two branches/block => 524,288 params/block.
    table = {
        "b1": {
            "trainable": 70_660_096,
            "dense": set(range(12, 16)),
            "nl_layers": set(range(0, 12)),
            "vlm_lora": 0,
            "conditioning": 6_651_904,
            "nonlinear": 6_291_456,
            "rank": 0,
        },
        "b2": {
            "trainable": 90_583_040,
            "dense": set(range(12, 16)),
            "nl_layers": set(range(0, 12)),
            "vlm_lora": 19_922_944,
            "conditioning": 6_651_904,
            "nonlinear": 6_291_456,
            "rank": 32,
        },
        "b3": {
            "trainable": 126_279_680,
            "dense": set(range(8, 16)),
            "nl_layers": set(range(0, 8)),
            "vlm_lora": 0,
            "conditioning": 6_651_904,
            "nonlinear": 4_194_304,
            "rank": 0,
        },
        "b4": {
            "trainable": 15_040_512,
            "dense": set(),
            "nl_layers": set(range(0, 16)),
            "vlm_lora": 0,
            "conditioning": 6_651_904,
            "nonlinear": 8_388_608,
            "rank": 0,
        },
    }
    return table[mode]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--unmerged", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--mode", choices=("b1", "b2", "b3", "b4"), required=True)
    args = ap.parse_args()
    exp = expected(args.mode)

    base = canonical_state(load_state(args.base))
    raw = canonical_state(load_state(args.unmerged))
    merged = canonical_state(load_state(args.merged))

    missing = [k for k in base if k not in raw]
    if missing:
        raise RuntimeError(f"Missing Base keys in unmerged checkpoint: {missing[:20]}")

    # Before LoRA merge, only selected ORIGINAL dense Flow blocks may alter Base tensors.
    bad_changed = []
    changed_dense = []
    dense_budget = 0
    for k, v in base.items():
        idx = flow_block_idx(k)
        if idx is not None and idx in exp["dense"]:
            dense_budget += v.numel()
        if torch.equal(v, raw[k]):
            continue
        if idx is not None and idx in exp["dense"]:
            changed_dense.append(k)
        else:
            bad_changed.append(k)
    if bad_changed:
        raise RuntimeError(f"Unexpected Base changes before merge: {bad_changed[:20]}")
    if exp["dense"] and not changed_dense:
        raise RuntimeError("No selected dense Flow tensor changed")

    extras = {k: v for k, v in raw.items() if k not in base}
    lora = {k: v for k, v in extras.items() if is_lora(k)}
    conditioning = {k: v for k, v in extras.items() if is_conditioning_adapter(k)}
    nonlinear = {k: v for k, v in extras.items() if is_nonlinear_adapter(k)}
    other = [k for k in extras if k not in lora and k not in conditioning and k not in nonlinear]
    if other:
        raise RuntimeError(f"Unexpected extra tensors in unmerged checkpoint: {other[:20]}")

    # LoRA must be text-only and only B2 carries it.
    vlm_lora = 0
    for k, v in lora.items():
        prefix = k.rsplit(".lora_", 1)[0]
        if k.endswith(".lora_A") and int(v.shape[0]) != int(exp["rank"]):
            raise RuntimeError(f"Wrong LoRA rank: {k}, shape={tuple(v.shape)}, expected={exp['rank']}")
        if is_text_module_prefix(prefix):
            vlm_lora += v.numel()
        else:
            raise RuntimeError(f"LoRA outside intended VLM Text targets: {prefix}")

    conditioning_params = sum(v.numel() for v in conditioning.values())
    nonlinear_params = sum(v.numel() for v in nonlinear.values())
    if vlm_lora != exp["vlm_lora"]:
        raise RuntimeError(f"VLM LoRA params={vlm_lora:,}, expected={exp['vlm_lora']:,}")
    if conditioning_params != exp["conditioning"]:
        raise RuntimeError(
            f"Conditioning params={conditioning_params:,}, expected={exp['conditioning']:,}"
        )
    if nonlinear_params != exp["nonlinear"]:
        raise RuntimeError(f"Nonlinear params={nonlinear_params:,}, expected={exp['nonlinear']:,}")

    nl_seen = {nonlinear_block_idx(k) for k in nonlinear}
    nl_seen.discard(None)
    if nl_seen != exp["nl_layers"]:
        raise RuntimeError(
            f"Nonlinear adapter placement mismatch: got={sorted(nl_seen)}, expected={sorted(exp['nl_layers'])}"
        )
    # Each configured layer must have both Attention and FFN branches.
    for idx in exp["nl_layers"]:
        keys = [k for k in nonlinear if flow_block_idx(k) == idx]
        if not any(".attn_nonlinear_adapter." in k for k in keys):
            raise RuntimeError(f"Layer {idx} missing attention nonlinear adapter")
        if not any(".ffn_nonlinear_adapter." in k for k in keys):
            raise RuntimeError(f"Layer {idx} missing FFN nonlinear adapter")

    cond_ups = [v for k, v in conditioning.items() if k.endswith(".up.weight")]
    if not cond_ups or not any(torch.count_nonzero(x).item() > 0 for x in cond_ups):
        raise RuntimeError("Conditioning adapter up projections stayed all-zero")
    nl_ups = [v for k, v in nonlinear.items() if k.endswith(".up.weight")]
    if not nl_ups or not any(torch.count_nonzero(x).item() > 0 for x in nl_ups):
        raise RuntimeError("DiT nonlinear adapter up projections stayed all-zero")

    total_budget = dense_budget + vlm_lora + conditioning_params + nonlinear_params
    if total_budget != exp["trainable"]:
        raise RuntimeError(
            f"Task parameter budget mismatch: dense={dense_budget:,}, text_lora={vlm_lora:,}, "
            f"conditioning={conditioning_params:,}, nonlinear={nonlinear_params:,}, "
            f"total={total_budget:,}, expected={exp['trainable']:,}"
        )

    # Merged checkpoint: no LoRA tensors remain; both adapter families remain as extras.
    if any(is_lora(k) for k in merged):
        raise RuntimeError("Merged checkpoint still contains LoRA tensors")
    merged_extras = {k: v for k, v in merged.items() if k not in base}
    bad_extra = [k for k in merged_extras if not is_conditioning_adapter(k) and not is_nonlinear_adapter(k)]
    if bad_extra:
        raise RuntimeError(f"Unexpected merged extras: {bad_extra[:20]}")
    if sum(v.numel() for k, v in merged_extras.items() if is_conditioning_adapter(k)) != exp["conditioning"]:
        raise RuntimeError("Merged conditioning adapter parameter count changed")
    if sum(v.numel() for k, v in merged_extras.items() if is_nonlinear_adapter(k)) != exp["nonlinear"]:
        raise RuntimeError("Merged nonlinear adapter parameter count changed")

    # After Text-LoRA merge only dense blocks and intended VLM text weights may differ.
    bad_after = []
    vlm_merged_changes = 0
    for k, v in base.items():
        if k not in merged:
            raise RuntimeError(f"Merged checkpoint missing Base key: {k}")
        if torch.equal(v, merged[k]):
            continue
        idx = flow_block_idx(k)
        if idx is not None and idx in exp["dense"]:
            continue
        if exp["vlm_lora"] and k.endswith(".weight"):
            prefix = k[:-len(".weight")]
            if is_text_module_prefix(prefix):
                vlm_merged_changes += 1
                continue
        bad_after.append(k)
    if bad_after:
        raise RuntimeError(f"Unexpected Base changes after merge: {bad_after[:20]}")
    if exp["vlm_lora"] and vlm_merged_changes <= 0:
        raise RuntimeError("No VLM Text LoRA update survived merge")

    print(f"[OK] {args.mode.upper()} nonlinear-action checkpoint audit passed")
    print(f"  dense Base parameter budget : {dense_budget:,}")
    print(f"  VLM Text LoRA params        : {vlm_lora:,}")
    print(f"  conditioning adapter params : {conditioning_params:,}")
    print(f"  DiT nonlinear params        : {nonlinear_params:,}")
    print(f"  nonlinear layers            : {sorted(nl_seen)}")
    print(f"  total task-specific params  : {total_budget:,}")
    print(f"  changed dense Base tensors  : {len(changed_dense)}")
    print(f"  merged VLM LoRA changes     : {vlm_merged_changes}")


if __name__ == "__main__":
    main()
