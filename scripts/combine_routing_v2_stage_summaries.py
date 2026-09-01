#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage-dir', type=Path, required=True)
    ap.add_argument('--tasks', nargs='+', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args()
    rows=[]
    for t in a.tasks:
        p=a.stage_dir/f'T{t}'/'suites'/'libero_goal'/'per_task_summary.csv'
        if not p.is_file(): raise FileNotFoundError(p)
        with p.open('r',encoding='utf-8-sig',newline='') as f:
            data=list(csv.DictReader(f))
        matches=[r for r in data if int(r['task_id'])==t]
        if len(matches)!=1: raise RuntimeError(f'{p}: expected one row for task {t}, got {len(matches)}')
        rows.append(matches[0])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=['task_id','task_description','task_name','successes','trials','success_rate']
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})
    print(f'[RoutingV2] combined stage summary -> {a.output}')
    print(a.output.read_text(encoding='utf-8').rstrip())
if __name__=='__main__': main()
