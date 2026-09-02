#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path


def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',action='append',required=True,help='DELTA=/path/to/run')
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; per_task=[]
    for spec in a.run:
        delta_s,root_s=spec.split('=',1); root=Path(root_s).resolve()
        metrics=read_json(root/'metrics'/'metrics.json')['metrics_percent']
        stages=read_csv(root/'routing_summary_all_stages.csv')
        cl4=read_csv(root/'CL4'/'routing_summary'/'routing_summary.csv')
        cl4_all=next(r for r in cl4 if r['scope']=='ALL')
        rows.append({
            'delta':float(delta_s),'run_root':str(root),
            'FWT_acquisition':metrics['FWT_acquisition_diagonal'],
            'BWT_final':metrics['BWT_final'],'NBT_REGEN':metrics['NBT_REGEN'],
            'AUC_CL':metrics['AUC_CL_lifecycle'],'final_seen_SR':metrics['final_seen_SR'],
            'CL4_semantic_acc':100*float(cl4_all['semantic_top1_accuracy']),
            'CL4_routing_acc':100*float(cl4_all['routing_accuracy']),
            'CL4_routing_gain_pp':float(cl4_all['routing_gain_vs_semantic_pp']),
            'CL4_gate_rate':100*float(cl4_all['gate_activation_rate']),
            'CL4_error_capture':100*float(cl4_all['semantic_error_capture_rate']) if cl4_all['semantic_error_capture_rate'] not in ('','None') else '',
            'CL4_recovered':int(cl4_all['fusion_recovered']),
            'CL4_damaged':int(cl4_all['fusion_damaged']),
            'CL4_net':int(cl4_all['fusion_net_corrections']),
            'CL4_mean_lambda_active':float(cl4_all['mean_lambda_activated']),
        })
        for r in cl4:
            if r['scope']=='ALL': continue
            per_task.append({
                'delta':float(delta_s),'task':r['scope'],
                'semantic_acc':100*float(r['semantic_top1_accuracy']),
                'routing_acc':100*float(r['routing_accuracy']),
                'gain_pp':float(r['routing_gain_vs_semantic_pp']),
                'gate_rate':100*float(r['gate_activation_rate']),
                'recovered':int(r['fusion_recovered']),
                'damaged':int(r['fusion_damaged']),
                'net':int(r['fusion_net_corrections']),
            })
    with (a.output_dir/'delta_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.output_dir/'delta_comparison_cl4_per_task.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_task[0]));w.writeheader();w.writerows(per_task)
    (a.output_dir/'delta_comparison.json').write_text(json.dumps({'runs':rows,'cl4_per_task':per_task},indent=2),encoding='utf-8')
    print('[RoutingV2][B2] delta comparison')
    for r in rows: print(r)
    print('output:',a.output_dir)
if __name__=='__main__': main()
