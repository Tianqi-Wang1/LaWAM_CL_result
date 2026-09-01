#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Offline temporal-routing sweep on an existing Routing-V1 chunk log.'
    )
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=0.8)
    p.add_argument('--normalization', choices=['none', 'candidate_mean'], default='candidate_mean')
    p.add_argument('--betas', type=str, default='0,0.3,0.5,0.6,0.8,0.9')
    p.add_argument('--margins', type=str, default='0,0.03,0.08')
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows=[]
    with path.open('r', encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f'Failed to parse {path}:{i}: {exc}') from exc
    if not rows:
        raise RuntimeError(f'Empty routing log: {path}')
    return rows


def instant_scores(row: dict, alpha: float, normalization: str) -> tuple[list[str], list[float]]:
    labels=list(row.get('candidates', {}).keys())
    if not labels:
        raise RuntimeError('Routing row has no candidates')
    dz=[float(row['candidates'][k]['dz']) for k in labels]
    dh=[float(row['candidates'][k]['dh']) for k in labels]
    if normalization == 'candidate_mean':
        mdz=max(sum(dz)/len(dz), 1e-8)
        mdh=max(sum(dh)/len(dh), 1e-8)
        dz=[x/mdz for x in dz]
        dh=[x/mdh for x in dh]
    return labels, [alpha*z + (1-alpha)*h for z,h in zip(dz,dh)]


def simulate(rows: list[dict], *, alpha: float, normalization: str, beta: float, margin: float) -> dict:
    episodes=defaultdict(list)
    for row in rows:
        context=str(row.get('context') or '')
        episode_id=row.get('episode_id')
        if episode_id is None:
            raise RuntimeError('Offline temporal analysis requires episode_id in every row')
        episodes[(context,str(episode_id))].append(row)

    total=correct=0
    switches=transitions=0
    instant_switches=0
    instant_correct=0
    by_task=defaultdict(lambda:[0,0])
    by_task_instant=defaultdict(lambda:[0,0])
    for _, ep_rows in episodes.items():
        ep_rows=sorted(ep_rows, key=lambda r:(int(r.get('chunk_index',0)),int(r.get('request_index',0))))
        ema=None
        current_idx=None
        prev_instant=None
        canonical_labels=None
        for row in ep_rows:
            labels,score=instant_scores(row,alpha,normalization)
            if canonical_labels is None:
                canonical_labels=labels
            elif labels != canonical_labels:
                raise RuntimeError(f'Candidate order changed within episode: {canonical_labels} -> {labels}')
            instant_idx=min(range(len(score)), key=lambda i:score[i])
            if ema is None:
                ema=list(score)
            else:
                ema=[beta*a+(1-beta)*b for a,b in zip(ema,score)]
            proposed=min(range(len(ema)), key=lambda i:ema[i])
            selected=proposed
            if margin>0 and current_idx is not None and proposed != current_idx:
                improvement=ema[current_idx]-ema[proposed]
                if improvement < margin:
                    selected=current_idx
            if current_idx is not None:
                transitions += 1
                switches += int(selected != current_idx)
                instant_switches += int(instant_idx != prev_instant)
            current_idx=selected
            prev_instant=instant_idx
            expected=row.get('expected_expert')
            if expected in labels:
                total += 1
                correct += int(labels[selected] == expected)
                instant_correct += int(labels[instant_idx] == expected)
                task=int(row['gt_task_id'])
                by_task[task][0]+=int(labels[selected]==expected); by_task[task][1]+=1
                by_task_instant[task][0]+=int(labels[instant_idx]==expected); by_task_instant[task][1]+=1
    return {
        'beta':beta,
        'margin':margin,
        'mode':'ema' if margin==0 else 'ema_hysteresis',
        'instant_accuracy':instant_correct/total if total else math.nan,
        'routing_accuracy':correct/total if total else math.nan,
        'accuracy_gain':(correct-instant_correct)/total if total else math.nan,
        'instant_switch_rate':instant_switches/transitions if transitions else 0.0,
        'switch_rate':switches/transitions if transitions else 0.0,
        'num_decisions':total,
        'per_task':{t:(a/b if b else math.nan) for t,(a,b) in sorted(by_task.items())},
    }


def main() -> None:
    a=parse_args()
    rows=load_rows(a.log)
    betas=parse_float_list(a.betas)
    margins=parse_float_list(a.margins)
    configs=[]
    for beta in betas:
        if beta < 0 or beta >= 1:
            raise ValueError(f'beta out of range: {beta}')
        for margin in margins:
            if margin < 0:
                raise ValueError(f'margin out of range: {margin}')
            configs.append(simulate(rows,alpha=a.alpha,normalization=a.normalization,beta=beta,margin=margin))
    configs.sort(key=lambda x:(-x['routing_accuracy'],x['switch_rate']))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    out=a.output_dir/'temporal_sweep.csv'
    tasks=sorted({t for x in configs for t in x['per_task']})
    fields=['rank','beta','margin','mode','instant_accuracy','routing_accuracy','accuracy_gain','instant_switch_rate','switch_rate','num_decisions',*[f'task_{t}_acc' for t in tasks]]
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for rank,x in enumerate(configs,1):
            row={k:v for k,v in x.items() if k!='per_task'}
            row['rank']=rank
            for t in tasks: row[f'task_{t}_acc']=x['per_task'].get(t,math.nan)
            w.writerow(row)
    summary={
        'alpha':a.alpha,
        'normalization':a.normalization,
        'best':configs[0],
        'output_csv':str(out),
    }
    (a.output_dir/'temporal_sweep_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True),encoding='utf-8')
    print('[RoutingV1 temporal offline sweep]')
    print(f'  alpha/norm: {a.alpha}/{a.normalization}')
    for i,x in enumerate(configs[:10],1):
        print(f"  #{i}: beta={x['beta']:.2f} margin={x['margin']:.3f} acc={x['routing_accuracy']:.4f} switch={x['switch_rate']:.4f} gain={x['accuracy_gain']:+.4f}")
    print(f'  csv: {out}')

if __name__=='__main__':
    main()
