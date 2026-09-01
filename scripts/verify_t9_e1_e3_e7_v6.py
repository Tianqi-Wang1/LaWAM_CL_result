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


def is_text_module_prefix(prefix: str) -> bool:
    if not prefix.startswith("policy_backend.vlm."):
        return False
    if ".language_model.layers." not in prefix:
        return False
    return any(prefix.endswith(s) for s in TEXT_SUFFIXES)


def is_flow_lora_prefix(prefix: str) -> bool:
    if not re.match(r"^policy_backend\.flow\.DiT\.transformer_blocks\.\d+\.", prefix):
        return False
    return any(
        token in prefix
        for token in (".attn1.to_q", ".attn1.to_k", ".attn1.to_v", ".attn1.to_out", ".ff.net.")
    )


def is_adapter(k: str) -> bool:
    return k.startswith("policy_backend.flow.") and ".conditioning_adapter." in k


def expected(mode: str):
    if mode == "e1":
        return {"trainable": 135_356_416, "dense": set(range(8,16)), "vlm_lora": 19_922_944, "flow_lora": 0, "adapter": 0}
    if mode == "e3":
        return {"trainable": 29_229_056, "dense": set(), "vlm_lora": 19_922_944, "flow_lora": 9_306_112, "adapter": 0}
    if mode == "e7":
        return {"trainable": 26_574_848, "dense": set(), "vlm_lora": 19_922_944, "flow_lora": 0, "adapter": 6_651_904}
    raise ValueError(mode)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--unmerged", type=Path, required=True)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--mode", choices=("e1","e3","e7"), required=True)
    ap.add_argument("--rank", type=int, default=32)
    args=ap.parse_args()
    exp=expected(args.mode)

    base=canonical_state(load_state(args.base))
    raw=canonical_state(load_state(args.unmerged))
    merged=canonical_state(load_state(args.merged))

    # Every Base key must survive.
    missing=[k for k in base if k not in raw]
    if missing: raise RuntimeError(f"Missing Base keys in unmerged: {missing[:20]}")

    # Before merge, only E1 dense blocks may alter Base tensors.
    bad_changed=[]
    changed_dense=[]
    for k,v in base.items():
        eq=torch.equal(v,raw[k])
        idx=flow_block_idx(k)
        if not eq:
            if idx is not None and idx in exp["dense"]:
                changed_dense.append(k)
            else:
                bad_changed.append(k)
    if bad_changed:
        raise RuntimeError(f"Unexpected Base changes before merge: {bad_changed[:20]}")
    if exp["dense"] and not changed_dense:
        raise RuntimeError("No selected dense Flow tensor changed")

    extras={k:v for k,v in raw.items() if k not in base}
    lora={k:v for k,v in extras.items() if is_lora(k)}
    adapters={k:v for k,v in extras.items() if is_adapter(k)}
    other=[k for k in extras if k not in lora and k not in adapters]
    if other: raise RuntimeError(f"Unexpected extra tensors: {other[:20]}")

    vlm_lora=0; flow_lora=0
    for k,v in lora.items():
        prefix=k.rsplit(".lora_",1)[0]
        if k.endswith(".lora_A") and int(v.shape[0]) != int(args.rank):
            raise RuntimeError(f"Wrong LoRA rank: {k} shape={tuple(v.shape)}")
        if is_text_module_prefix(prefix): vlm_lora += v.numel()
        elif is_flow_lora_prefix(prefix): flow_lora += v.numel()
        else: raise RuntimeError(f"LoRA outside intended targets: {prefix}")

    adapter_params=sum(v.numel() for v in adapters.values())
    if vlm_lora != exp["vlm_lora"]:
        raise RuntimeError(f"VLM LoRA params={vlm_lora:,}, expected={exp['vlm_lora']:,}")
    if flow_lora != exp["flow_lora"]:
        raise RuntimeError(f"Flow LoRA params={flow_lora:,}, expected={exp['flow_lora']:,}")
    if adapter_params != exp["adapter"]:
        raise RuntimeError(f"Adapter params={adapter_params:,}, expected={exp['adapter']:,}")
    if args.mode == "e7":
        ups=[v for k,v in adapters.items() if k.endswith(".up.weight")]
        if not ups or not any(torch.count_nonzero(x).item() > 0 for x in ups):
            raise RuntimeError("E7 adapter up projections stayed all-zero")

    # Merged: no LoRA remains. E7 keeps adapter tensors as extras.
    if any(is_lora(k) for k in merged): raise RuntimeError("Merged checkpoint still has LoRA tensors")
    merged_extras={k:v for k,v in merged.items() if k not in base}
    if args.mode == "e7":
        bad=[k for k in merged_extras if not is_adapter(k)]
        if bad: raise RuntimeError(f"Unexpected merged extras: {bad[:20]}")
        if sum(v.numel() for v in merged_extras.values()) != exp["adapter"]:
            raise RuntimeError("Merged E7 adapter parameter count changed")
    elif merged_extras:
        raise RuntimeError(f"Unexpected merged extras: {list(merged_extras)[:20]}")

    # After merge, allow selected dense blocks and intended LoRA base weights only.
    bad_after=[]; vlm_merged_changes=0; flow_merged_changes=0
    for k,v in base.items():
        if k not in merged:
            raise RuntimeError(f"Merged checkpoint missing Base key {k}")
        if torch.equal(v, merged[k]):
            continue
        idx=flow_block_idx(k)
        if idx is not None and idx in exp["dense"]:
            continue
        if k.endswith(".weight"):
            prefix=k[:-len(".weight")]
            if is_text_module_prefix(prefix):
                vlm_merged_changes += 1; continue
            if is_flow_lora_prefix(prefix) and exp["flow_lora"] > 0:
                flow_merged_changes += 1; continue
        bad_after.append(k)
    if bad_after: raise RuntimeError(f"Unexpected changes after merge: {bad_after[:20]}")
    if vlm_merged_changes <= 0: raise RuntimeError("No VLM text LoRA update survived merge")
    if exp["flow_lora"] > 0 and flow_merged_changes <= 0: raise RuntimeError("No Flow LoRA update survived merge")

    print(f"[OK] {args.mode.upper()} checkpoint audit passed")
    print(f"  canonical VLM LoRA params : {vlm_lora:,}")
    print(f"  canonical Flow LoRA params: {flow_lora:,}")
    print(f"  conditioning adapter params: {adapter_params:,}")
    print(f"  expected task trainable params: {exp['trainable']:,}")
    print(f"  dense Base tensors changed pre-merge: {len(changed_dense)}")
    print(f"  merged VLM/Flow target changes: {vlm_merged_changes}/{flow_merged_changes}")


if __name__ == "__main__":
    main()
