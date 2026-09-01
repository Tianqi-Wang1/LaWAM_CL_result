#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read_rows(root: Path):
    paths = sorted(root.glob("T*/dynamics_probe.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No T*/dynamics_probe.jsonl under {root}")
    rows = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line); r["_source_file"] = str(p); rows.append(r)
    if not rows:
        raise RuntimeError("No dynamics probe rows")
    return rows, paths


def quantile(vals, q):
    vals = sorted(float(x) for x in vals)
    if not vals: return math.nan
    if len(vals) == 1: return vals[0]
    x = (len(vals)-1)*q; lo=int(math.floor(x)); hi=int(math.ceil(x))
    if lo == hi: return vals[lo]
    w=x-lo; return vals[lo]*(1-w)+vals[hi]*w


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def threshold_sweep(rows: list[dict], metric: str, points: int = 101):
    vals = [float(r[metric]) for r in rows]
    thresholds = sorted(set([min(vals)-1e-12] + [quantile(vals, i/(points-1)) for i in range(points)] + [max(vals)+1e-12]))
    sem_acc = sum(bool(r["semantic_top1_correct"]) for r in rows)/len(rows)
    n_sem_err = sum(not bool(r["semantic_top1_correct"]) for r in rows)
    out=[]
    for thr in thresholds:
        pred=[]; gated=[]
        for r in rows:
            use_dyn = float(r[metric]) < thr
            gated.append(use_dyn)
            pred.append(int(r["dynamics_winner_task"] if use_dyn else r["semantic_top1_task"]))
        correct = [p == int(r["gt_task_id"]) for p,r in zip(pred,rows)]
        gated_rows=[r for r,g in zip(rows,gated) if g]
        sem_err_gated=sum(not bool(r["semantic_top1_correct"]) for r in gated_rows)
        recovered=sum(bool(r["semantic_error_recovered"]) for r in gated_rows)
        damaged=sum(bool(r["semantic_correct_damaged"]) for r in gated_rows)
        out.append({
            "metric":metric,"threshold":thr,"gate_rule":f"{metric} < threshold",
            "activation_count":sum(gated),"activation_rate":sum(gated)/len(rows),
            "hybrid_correct":sum(correct),"hybrid_accuracy":sum(correct)/len(rows),
            "semantic_baseline_accuracy":sem_acc,"gain_vs_semantic":sum(correct)/len(rows)-sem_acc,
            "semantic_errors_total":n_sem_err,"semantic_errors_gated":sem_err_gated,
            "semantic_error_capture_rate":sem_err_gated/n_sem_err if n_sem_err else 1.0,
            "recovered_errors_in_gate":recovered,"damaged_correct_in_gate":damaged,
            "net_corrections":recovered-damaged,
        })
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--output-dir", default=None); ap.add_argument("--sweep-points", type=int, default=101); args=ap.parse_args()
    root=Path(args.root).expanduser().resolve(); out=Path(args.output_dir).expanduser().resolve() if args.output_dir else root/"dynamics_summary"; out.mkdir(parents=True,exist_ok=True)
    rows, paths=read_rows(root); n=len(rows)
    gt_tasks=sorted(set(int(r["gt_task_id"]) for r in rows))
    sem_correct=sum(bool(r["semantic_top1_correct"]) for r in rows); top2=sum(bool(r["semantic_top2_correct"]) for r in rows); dyn_correct=sum(bool(r["dynamics_correct"]) for r in rows)
    sem_err=[r for r in rows if not bool(r["semantic_top1_correct"])]; sem_ok=[r for r in rows if bool(r["semantic_top1_correct"])]
    recovered=sum(bool(r["semantic_error_recovered"]) for r in sem_err); damaged=sum(bool(r["semantic_correct_damaged"]) for r in sem_ok)

    task_rows=[]
    for t in gt_tasks:
        rr=[r for r in rows if int(r["gt_task_id"])==t]; err=[r for r in rr if not bool(r["semantic_top1_correct"])]; ok=[r for r in rr if bool(r["semantic_top1_correct"])]
        task_rows.append({
            "gt_task_id":t,"n":len(rr),
            "semantic_top1_accuracy":sum(bool(r["semantic_top1_correct"]) for r in rr)/len(rr),
            "semantic_top2_recall":sum(bool(r["semantic_top2_correct"]) for r in rr)/len(rr),
            "dynamics_top2_accuracy":sum(bool(r["dynamics_correct"]) for r in rr)/len(rr),
            "semantic_errors":len(err),"recovered_errors":sum(bool(r["semantic_error_recovered"]) for r in err),
            "recovery_rate":sum(bool(r["semantic_error_recovered"]) for r in err)/len(err) if err else math.nan,
            "semantic_correct":len(ok),"damaged_correct":sum(bool(r["semantic_correct_damaged"]) for r in ok),
            "damage_rate":sum(bool(r["semantic_correct_damaged"]) for r in ok)/len(ok) if ok else math.nan,
        })
    write_csv(out/"dynamics_per_task.csv",task_rows)

    # Confusion for Dynamics winner.
    preds=sorted(set(gt_tasks)|set(int(r["dynamics_winner_task"]) for r in rows)); conf={t:Counter() for t in gt_tasks}
    for r in rows: conf[int(r["gt_task_id"])][int(r["dynamics_winner_task"])]+=1
    conf_rows=[]
    for g in gt_tasks:
        d={"gt_task_id":g}; d.update({f"pred_t{p}":conf[g][p] for p in preds}); conf_rows.append(d)
    write_csv(out/"dynamics_top2_confusion.csv",conf_rows)

    metrics=["semantic_normalized_gap","semantic_relative_margin","semantic_abs_margin","semantic_error_ratio"]
    all_sweep=[]; best={}
    for metric in metrics:
        sw=threshold_sweep(rows,metric,max(11,args.sweep_points)); all_sweep.extend(sw)
        # Diagnostic test-set oracle threshold: max accuracy, then min activation.
        b=max(sw,key=lambda r:(float(r["hybrid_accuracy"]),-float(r["activation_rate"])))
        best[metric]=b
    write_csv(out/"hybrid_confidence_threshold_sweep.csv",all_sweep)

    # Confidence deciles for primary normalized gap: does Dynamics help mostly at low confidence?
    vals=[float(r["semantic_normalized_gap"]) for r in rows]; edges=[quantile(vals,i/10) for i in range(11)]; bins=[]
    for b in range(10):
        lo,hi=edges[b],edges[b+1]
        rr=[r for r in rows if float(r["semantic_normalized_gap"])>=lo and (float(r["semantic_normalized_gap"])<=hi if b==9 else float(r["semantic_normalized_gap"])<hi)]
        if not rr: continue
        sem=sum(bool(r["semantic_top1_correct"]) for r in rr)/len(rr); dyn=sum(bool(r["dynamics_correct"]) for r in rr)/len(rr)
        rec=sum(bool(r["semantic_error_recovered"]) for r in rr); dmg=sum(bool(r["semantic_correct_damaged"]) for r in rr)
        bins.append({"bin":b,"lo":lo,"hi":hi,"n":len(rr),"semantic_accuracy":sem,"dynamics_accuracy":dyn,"dynamics_minus_semantic":dyn-sem,"recovered":rec,"damaged":dmg,"net_corrections":rec-dmg})
    write_csv(out/"confidence_dynamics_decile_bins.csv",bins)

    summary={
        "root":str(root),"log_files":[str(p) for p in paths],"num_decisions":n,
        "semantic_top1_accuracy":sem_correct/n,"semantic_top2_recall":top2/n,
        "dynamics_top2_accuracy_always_on":dyn_correct/n,
        "semantic_error_count":len(sem_err),"recovered_error_count":recovered,"recovery_rate":recovered/len(sem_err) if sem_err else 1.0,
        "semantic_correct_count":len(sem_ok),"damaged_correct_count":damaged,"damage_rate":damaged/len(sem_ok) if sem_ok else 0.0,
        "always_on_net_gain":dyn_correct/n-sem_correct/n,
        "primary_confidence_metric":"semantic_normalized_gap=(e2-e1)/e2; lower means more ambiguous",
        "diagnostic_best_thresholds":best,
        "warning":"Best thresholds use GT labels from this passive probe and are diagnostic only. Freeze the final gate on a held-out routing validation split, not the final test set.",
    }
    (out/"dynamics_probe_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    print("======================================================================")
    print(" Routing-V2 passive Dynamics verification")
    print(f" Decisions            : {n}")
    print(f" Semantic Top1        : {100*summary['semantic_top1_accuracy']:.2f}%")
    print(f" Semantic Top2 recall : {100*summary['semantic_top2_recall']:.2f}%")
    print(f" Dynamics Top2        : {100*summary['dynamics_top2_accuracy_always_on']:.2f}%")
    print(f" Recovery             : {recovered}/{len(sem_err)} = {100*summary['recovery_rate']:.2f}%")
    print(f" Damage               : {damaged}/{len(sem_ok)} = {100*summary['damage_rate']:.2f}%")
    print("---------------------------------------------------------------------")
    b=best["semantic_normalized_gap"]
    print(f" Diagnostic gated best: threshold<{float(b['threshold']):.6g}, activate={100*float(b['activation_rate']):.2f}%, hybrid={100*float(b['hybrid_accuracy']):.2f}%, gain={100*float(b['gain_vs_semantic']):+.2f}pp")
    print(f" Output               : {out}")
    print("======================================================================")


if __name__=="__main__": main()
