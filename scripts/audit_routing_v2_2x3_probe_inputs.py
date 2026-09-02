#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

VARIANTS = (
    "taskwm_hdhz", "taskwm_hdh", "taskwm_dh",
    "basewm_hdhz", "basewm_hdh", "basewm_dh",
)


def load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_mode(state: dict) -> str:
    iw = state["input_proj.weight"]
    dw = state["delta_decoder.weight"]
    d = int(dw.shape[0]); inp = int(iw.shape[1])
    has_z = any(k.startswith("z_decoder.") for k in state)
    if has_z:
        return "hdhz"
    if inp == 2*d:
        return "hdh"
    if inp == d:
        return "dh"
    return f"unknown({inp})"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v2-root", required=True)
    args=ap.parse_args()
    root=Path(args.v2_root).expanduser().resolve()
    rows=[]
    for t in (6,7,8,9):
        for v in VARIANTS:
            p=root/f"task{t}"/"routing_memory_variants"/v/"dynamics_ae.pt"
            if not p.is_file():
                raise FileNotFoundError(p)
            st=load(p)
            mode=infer_mode(st)
            expected=v.split("_",1)[1]
            if mode != expected:
                raise RuntimeError(f"{p}: mode={mode}, expected={expected}")
            meta=p.with_suffix(".json")
            m=json.loads(meta.read_text()) if meta.is_file() else {}
            rows.append((t,v,mode,sum(x.numel() for x in st.values() if torch.is_tensor(x)),m.get("wm_source","?")))
    print("task variant          input  params      wm_meta")
    for t,v,m,n,w in rows:
        print(f"T{t:<2}  {v:<16} {m:<5} {n:>10,}  {w}")
    print(f"[OK] audited {len(rows)} Dynamics memories")

if __name__=='__main__': main()
