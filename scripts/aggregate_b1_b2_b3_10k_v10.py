#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path('/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seq_single_task_nonlinear_10k_v10')
OLD = Path('/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seq_single_task_nonlinear_v9')
MODES = ('b1','b2','b3')

def load(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

rows=[]
for mode in MODES:
    p=ROOT/mode/'seq_single_task_summary.csv'
    if not p.is_file():
        raise FileNotFoundError(p)
    for r in load(p):
        rows.append(r)

out=ROOT/'b1_b2_b3_10k_summary.csv'
fields=['mode','stage','task_id','task_description','successes','trials','success_rate','task_specific_params','summary_file']
with out.open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

print('\nB1/B2/B3 fresh single-task sweep @ 10k steps')
print('Mode | Params(M) | T6 | T7 | T8 | T9 | Mean')
print('-----+-----------+----+----+----+----+-----')
for mode in MODES:
    rs=[r for r in rows if r['mode']==mode]
    rs=sorted(rs, key=lambda x:int(x['task_id']))
    vals=[float(r['success_rate']) for r in rs]
    params=int(rs[0]['task_specific_params'])/1e6
    print(f'{mode:>4s} | {params:9.2f} | ' + ' | '.join(f'{v:.3f}' for v in vals) + f' | {sum(vals)/len(vals):.3f}')
print(f'[OK] Saved: {out}')

# Optional comparison against the existing 2k v9 summaries.
if all((OLD/m/'seq_single_task_summary.csv').is_file() for m in MODES):
    comp=[]
    print('\n2k -> 10k comparison')
    print('Mode | Mean@2k | Mean@10k | Delta')
    print('-----+---------+----------+------')
    for mode in MODES:
        old=load(OLD/mode/'seq_single_task_summary.csv')
        new=load(ROOT/mode/'seq_single_task_summary.csv')
        om=sum(float(r['success_rate']) for r in old)/len(old)
        nm=sum(float(r['success_rate']) for r in new)/len(new)
        print(f'{mode:>4s} | {om:7.3f} | {nm:8.3f} | {nm-om:+.3f}')
        by_old={int(r['task_id']):float(r['success_rate']) for r in old}
        by_new={int(r['task_id']):float(r['success_rate']) for r in new}
        for t in sorted(by_new):
            comp.append({'mode':mode,'task_id':t,'sr_2k':by_old.get(t,''),'sr_10k':by_new[t], 'delta': '' if t not in by_old else by_new[t]-by_old[t]})
    cout=ROOT/'compare_2k_vs_10k.csv'
    with cout.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['mode','task_id','sr_2k','sr_10k','delta']); w.writeheader(); w.writerows(comp)
    print(f'[OK] Saved comparison: {cout}')
