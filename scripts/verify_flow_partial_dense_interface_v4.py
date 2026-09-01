#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch

CANONICAL="policy_backend.flow."
ALIAS="policy_action_head."
INTERFACE_PREFIXES=(
    "enc_vlm.", "action_encoder.", "action_decoder.", "enc_state.",
    "DiT.proj_out_1.", "DiT.proj_out_2.", "DiT.timestep_encoder.",
)


def load_state(path: Path):
    try:
        obj=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try: obj=torch.load(path,map_location="cpu",weights_only=True)
        except TypeError: obj=torch.load(path,map_location="cpu")
    if isinstance(obj,dict):
        for w in ("state_dict","model","module"):
            n=obj.get(w)
            if isinstance(n,dict) and n and any(torch.is_tensor(v) for v in n.values()):
                obj=n; break
    if not isinstance(obj,dict): raise RuntimeError(f"Unsupported checkpoint {path}")
    return obj


def eq(a,b):
    return torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)


def is_flow(k): return k.startswith(CANONICAL) or k.startswith(ALIAS)


def flow_map(s):
    out={}; suffixes=set()
    for k in s:
        if k.startswith(CANONICAL): suffixes.add(k[len(CANONICAL):])
        elif k.startswith(ALIAS): suffixes.add(k[len(ALIAS):])
    for suf in sorted(suffixes):
        c=CANONICAL+suf; a=ALIAS+suf
        if c in s and a in s and torch.is_tensor(s[c]) and torch.is_tensor(s[a]) and not eq(s[c],s[a]):
            raise RuntimeError(f"Canonical/alias mismatch: {suf}")
        if c in s and torch.is_tensor(s[c]): out[suf]=s[c]
        elif a in s and torch.is_tensor(s[a]): out[suf]=s[a]
    return out


def parse_layers(t):
    x=tuple(int(z.strip()) for z in t.split(',') if z.strip())
    if not x or len(x)!=len(set(x)): raise ValueError(f"Bad layers: {x}")
    return x


def allowed_base_suffix(k,layers,interface):
    if any(k.startswith(f"DiT.transformer_blocks.{i}.") for i in layers): return True
    if interface in ("dense","merged_lora") and k.startswith(INTERFACE_PREFIXES): return True
    return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-ckpt',type=Path,required=True)
    ap.add_argument('--final-ckpt',type=Path,required=True)
    ap.add_argument('--layers',required=True)
    ap.add_argument('--interface-mode',choices=['none','dense','lora','merged_lora'],default='none')
    args=ap.parse_args()
    layers=parse_layers(args.layers)
    base=load_state(args.base_ckpt); final=load_state(args.final_ckpt)

    # Upstream exact.
    checked=0; changed=[]
    for k,v in base.items():
        if not torch.is_tensor(v) or is_flow(k): continue
        checked+=1
        if k not in final or not eq(v,final[k]): changed.append(k)
    if changed: raise RuntimeError(f"Frozen upstream changed: {changed[:30]}")

    b=flow_map(base); f=flow_map(final)
    residual=[k for k in f if 'residual_expert_blocks.' in k]
    if residual: raise RuntimeError(f"Residual tensors unexpectedly present: {residual[:20]}")

    lora=[k for k in f if k.endswith('.lora_A') or k.endswith('.lora_B')]
    if args.interface_mode=='lora':
        if not lora: raise RuntimeError('Expected interface LoRA tensors, found none')
        bad=[k for k in lora if not k.startswith(INTERFACE_PREFIXES)]
        if bad: raise RuntimeError(f"LoRA outside interface: {bad[:20]}")
        bkeys=[k for k in lora if k.endswith('.lora_B')]
        if not any(torch.count_nonzero(f[k]).item()>0 for k in bkeys):
            raise RuntimeError('All interface lora_B tensors remain zero')
    else:
        if lora: raise RuntimeError(f"Unexpected LoRA tensors: {lora[:20]}")

    # Every Base Flow key must still exist; LoRA mode may add adapter keys.
    missing=sorted(set(b)-set(f))
    if missing: raise RuntimeError(f"Missing Base Flow keys: {missing[:20]}")

    changed_base=[]; outside=[]
    for k,v in b.items():
        if not eq(v,f[k]):
            changed_base.append(k)
            if not allowed_base_suffix(k,layers,args.interface_mode): outside.append(k)
    if outside: raise RuntimeError(f"Base Flow params changed outside allowed set: {outside[:30]}")

    # Selected dense blocks must change.
    per={}
    for i in layers:
        ks=[k for k in b if k.startswith(f"DiT.transformer_blocks.{i}.")]
        ch=[k for k in ks if not eq(b[k],f[k])]
        if not ch: raise RuntimeError(f"Dense layer {i} has no changed tensors")
        per[i]=(len(ks),len(ch),sum(b[k].numel() for k in ks))

    if args.interface_mode in ('dense','merged_lora'):
        ik=[k for k in b if k.startswith(INTERFACE_PREFIXES)]
        ich=[k for k in ik if not eq(b[k],f[k])]
        if not ich: raise RuntimeError(f"Interface mode {args.interface_mode} but no interface Base tensors changed")
    elif args.interface_mode=='lora':
        # Base interface must be bitwise exact before merge.
        bad=[k for k in b if k.startswith(INTERFACE_PREFIXES) and not eq(b[k],f[k])]
        if bad: raise RuntimeError(f"Interface Base weights changed before LoRA merge: {bad[:20]}")

    print('[OK] Partial-dense/interface checkpoint verified')
    print(f'  layers                 : {layers}')
    print(f'  interface_mode         : {args.interface_mode}')
    print(f'  upstream_exact_tensors : {checked}')
    print(f'  changed_base_flow      : {len(changed_base)}')
    print(f'  lora_tensors           : {len(lora)}')
    print(f'  per_layer              : {per}')


if __name__=='__main__': main()
