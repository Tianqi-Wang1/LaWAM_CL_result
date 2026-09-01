#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv
from pathlib import Path

STAGES=[('CL1',[6]),('CL2',[6,7]),('CL3',[6,7,8]),('CL4',[6,7,8,9])]
TASKS=[6,7,8,9]

def read_summary(path:Path):
    out={}
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): out[int(r['task_id'])]=float(r['success_rate'])
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=['stage']+[f'task_{t}' for t in TASKS]
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for stage,seen in STAGES:
            p=a.root/stage/'per_task_summary.csv'
            if not p.is_file(): raise FileNotFoundError(p)
            vals=read_summary(p)
            missing=[t for t in seen if t not in vals]
            if missing: raise RuntimeError(f'{p}: missing {missing}')
            w.writerow({'stage':stage, **{f'task_{t}':(f'{vals[t]:.8f}' if t in seen else '') for t in TASKS}})
    print(f'[RoutingV2] SR matrix -> {a.output}')
    print(a.output.read_text(encoding='utf-8').rstrip())
if __name__=='__main__': main()
