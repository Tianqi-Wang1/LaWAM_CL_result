#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path


def read_rows(paths):
    rows=[]
    for p in paths:
        with p.open('r',encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if line: rows.append(json.loads(line))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage-dir',type=Path,required=True); ap.add_argument('--tasks',nargs='+',type=int,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    paths=[a.stage_dir/f'T{t}'/'routing_chunks.jsonl' for t in a.tasks]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    rows=read_rows(paths); n=len(rows)
    if n==0: raise RuntimeError('no routing rows')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for r in rows: groups[int(r['gt_task_id'])].append(r)
    summary=[]
    for key,subset in [('ALL',rows),*[(f'T{t}',groups[t]) for t in a.tasks]]:
        if not subset: continue
        m=len(subset); sem=sum(bool(r['semantic_top1_correct']) for r in subset); route=sum(bool(r['routing_correct']) for r in subset); gated=sum(bool(r['gate_active']) for r in subset)
        summary.append({'scope':key,'decisions':m,'semantic_top1_accuracy':sem/m,'routing_accuracy':route/m,'gate_activation_rate':gated/m,'mean_lambda_dyn':sum(float(r['lambda_dyn']) for r in subset)/m})
    with (a.output_dir/'routing_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    counts=Counter((int(r['gt_task_id']),int(r['selected_task'])) for r in rows)
    candidates=sorted({int(t) for r in rows for t in r['candidate_tasks']})
    with (a.output_dir/'routing_confusion.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['gt_task']+[f'pred_T{t}' for t in candidates]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for gt in a.tasks: w.writerow({'gt_task':gt,**{f'pred_T{t}':counts[(gt,t)] for t in candidates}})
    payload={'decisions':n,'summary':summary,'candidate_tasks':candidates}
    (a.output_dir/'routing_summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(f'[RoutingV2] routing summary -> {a.output_dir}')
    for r in summary: print(r)
if __name__=='__main__': main()
