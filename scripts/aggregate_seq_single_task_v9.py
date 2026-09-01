#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path

PARAMS = {"b1": 70_660_096, "b2": 90_583_040, "b3": 126_279_680}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('b1','b2','b3'), required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--tasks', type=int, nargs='+', default=[6,7,8,9])
    args=ap.parse_args()
    rows=[]
    for task in args.tasks:
        p=args.root / args.mode / f'task{task}' / 'latest_summary_path.txt'
        if not p.is_file():
            raise FileNotFoundError(f'Missing pointer: {p}')
        summary=Path(p.read_text(encoding='utf-8').strip())
        with summary.open(newline='',encoding='utf-8') as f:
            data=list(csv.DictReader(f))
        matched=[r for r in data if int(r['task_id'])==task]
        if len(matched)!=1:
            raise RuntimeError(f'T{task}: expected exactly one row in {summary}, got {len(matched)}')
        r=matched[0]
        rows.append({
            'mode':args.mode,
            'stage':f'CL{task-5}',
            'task_id':task,
            'task_description':r.get('task_description',''),
            'successes':int(r['successes']),
            'trials':int(r['trials']),
            'success_rate':float(r['success_rate']),
            'task_specific_params':PARAMS[args.mode],
            'summary_file':str(summary),
        })
    outdir=args.root/args.mode
    outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/'seq_single_task_summary.csv'
    fields=list(rows[0].keys())
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    mean=sum(r['success_rate'] for r in rows)/len(rows)
    total_s=sum(r['successes'] for r in rows); total_t=sum(r['trials'] for r in rows)
    print('\nSeq-task fresh single-task sweep')
    print('Mode | Stage | Task | Success | Trials | SR')
    print('-----+-------+------+---------+--------+------')
    for r in rows:
        print(f"{r['mode']:>4s} | {r['stage']:>5s} | T{r['task_id']}   | {r['successes']:>7d} | {r['trials']:>6d} | {r['success_rate']:.4f}")
    print(f'\nMean task SR: {mean:.4f}')
    print(f'Overall: {total_s}/{total_t} = {total_s/total_t:.4f}')
    print(f'[OK] Saved: {out}')

if __name__=='__main__': main()
