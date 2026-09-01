#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--config', default=None)
    args=ap.parse_args()
    sd=torch.load(args.checkpoint,map_location='cpu')
    prefix='policy_backend.routing_v2_memory.'
    mem={k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)}
    if not mem:
        raise RuntimeError(f'No {prefix} tensors found in {args.checkpoint}')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(mem,out)
    meta={'source_checkpoint':str(Path(args.checkpoint).resolve()),'num_tensors':len(mem),'num_params':sum(v.numel() for v in mem.values())}
    if args.config:
        meta['config']=str(Path(args.config).resolve())
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'[OK] Routing-V2 memory extracted: {out} tensors={len(mem)} params={meta["num_params"]:,}')

if __name__=='__main__': main()
