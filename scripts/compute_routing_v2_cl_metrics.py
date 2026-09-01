#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from statistics import fmean

TASKS=[6,7,8,9]
STAGES=['CL1','CL2','CL3','CL4']

def load(path:Path):
    with path.open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    M={}
    for r in rows:
        st=r['stage'].strip(); M[st]={}
        for t in TASKS:
            raw=str(r.get(f'task_{t}','')).strip(); M[st][t]=math.nan if not raw else float(raw)
    for i,(st,t) in enumerate(zip(STAGES,TASKS)):
        for later in STAGES[i:]:
            if later not in M or math.isnan(M[later][t]): raise RuntimeError(f'missing task_{t} at {later}')
    return M

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--eps',type=float,default=1e-8); a=ap.parse_args()
    M=load(a.matrix)
    diag=[M[st][t] for st,t in zip(STAGES,TASKS)]
    detail=[]
    nbt_vals=[]
    for i,t in enumerate(TASKS):
        vals=[M[st][t] for st in STAGES[i:]]
        intro=vals[0]
        nbt=None
        if len(vals)>1 and intro>a.eps:
            nbt=fmean((intro-x)/intro for x in vals[1:]); nbt_vals.append(nbt)
        detail.append({'task_id':t,'intro_stage':STAGES[i],'values':vals,'NBT':nbt,'AUC':fmean(vals)})
    # Compatibility with the user's existing REGEN-style engine: FWT is the
    # acquisition/diagonal SR because unseen-task pre-evaluation is not part of
    # this LIBERO protocol. Standard pre-learning FWT is therefore unavailable.
    fwt=fmean(diag)
    old_tasks=TASKS[:-1]
    bwt_final=fmean(M['CL4'][t]-M[STAGES[TASKS.index(t)]][t] for t in old_tasks)
    all_backward=[]
    for i,t in enumerate(TASKS[:-1]):
        intro=M[STAGES[i]][t]
        for st in STAGES[i+1:]: all_backward.append(M[st][t]-intro)
    bwt_lifecycle=fmean(all_backward) if all_backward else math.nan
    auc=fmean(d['AUC'] for d in detail)
    final=fmean(M['CL4'][t] for t in TASKS)
    metrics={
        'FWT':100*fwt,
        'FWT_acquisition_diagonal':100*fwt,
        'BWT_final':100*bwt_final,
        'BWT_lifecycle_mean':100*bwt_lifecycle,
        'Final_forgetting_pp':-100*bwt_final,
        'NBT_REGEN':100*fmean(nbt_vals) if nbt_vals else math.nan,
        'AUC_CL_lifecycle':100*auc,
        'final_seen_SR':100*final,
    }
    stage_avg={st:100*fmean(M[st][t] for t in TASKS[:idx+1]) for idx,st in enumerate(STAGES)}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/'metrics.json').write_text(json.dumps({
        'protocol':'routing_v2_cl_only',
        'metric_note':'FWT follows the existing acquisition/diagonal convention; standard unseen-task pre-learning FWT is not measured.',
        'metrics_percent':metrics,
        'stage_seen_average_percent':stage_avg,
    },indent=2),encoding='utf-8')
    with (a.output_dir/'task_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['task_id','intro_stage','intro_sr_percent','final_sr_percent','BWT_final_pp','NBT_REGEN_percent','AUC_percent','lifecycle_sr_percent'])
        for d in detail:
            vals=d['values']; bwt=100*(vals[-1]-vals[0]) if len(vals)>1 else ''
            w.writerow([d['task_id'],d['intro_stage'],100*vals[0],100*vals[-1],bwt,'' if d['NBT'] is None else 100*d['NBT'],100*d['AUC'],'|'.join(f'{100*x:.4f}' for x in vals)])
    with (a.output_dir/'stage_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['stage','seen_avg_sr_percent']);
        for st in STAGES: w.writerow([st,stage_avg[st]])
    print('[RoutingV2 CL metrics]')
    for k,v in metrics.items(): print(f'  {k}: {v:.4f}')
    print('  stage seen averages:', stage_avg)
    print(f'  output: {a.output_dir}')
if __name__=='__main__': main()
