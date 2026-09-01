#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the already validated trajectory-balanced Semantic/Conditioning analysis
# helpers in the same scripts/ directory.
import analyze_libero_semantic_conditioning as sem


STAGES = sem.STAGES
ROUTES = ["base_ref", "qformer_only", "upstream_only", "full"]
TEACHER_PREFIXES = (
    "policy_backend.lam.vision_encoder.",
    "policy_backend.lam.encoder.",
    "policy_backend.lam.vq.",
)
METRICS = (
    "teacher_rel_l2",
    "teacher_cosine",
    "teacher_mse",
    "teacher_rel_l2_excess_vs_base",
    "teacher_cosine_drop_vs_base",
    "z_drift_rel_l2",
    "z_drift_cosine",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="LaWAM latent-action forgetting with fixed teacher and 2x2 module intervention."
    )
    p.add_argument("--suite", required=True, choices=["libero_goal", "libero_object"])
    p.add_argument("--config-yaml", type=Path, default=Path("starVLA/config/training/train_libero.yaml"))
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-ids", nargs="+", type=int, default=[0,1,2,3,4,5])
    p.add_argument("--trajectory-fractions", nargs="+", type=float, default=[0.25,0.50,0.75])
    p.add_argument("--max-trajectories-per-task", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--split", choices=["all","train","val"], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--reuse-input-cache", action="store_true")
    p.add_argument("--skip-teacher-stability-check", action="store_true")
    p.add_argument("--allow-teacher-mismatch", action="store_true")
    return p.parse_args()


def load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state(state):
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k,v in state.items()}
    return state


def tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    if t.numel():
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def is_teacher_key(key: str) -> bool:
    return any(key.startswith(p) for p in TEACHER_PREFIXES)


def verify_teacher_stability(chain, out_csv: Path, allow_mismatch: bool):
    print("\n[teacher-check] hashing Base vision_encoder + IDM encoder + vq")
    base_state = normalize_state(load_state(chain["Base"]))
    base_hash = {
        k: tensor_digest(v) for k,v in base_state.items()
        if is_teacher_key(k) and torch.is_tensor(v)
    }
    del base_state
    gc.collect()
    if not base_hash:
        raise RuntimeError("No teacher-producing LAM tensors found in Base checkpoint.")

    rows = [dict(stage="Base", checked_keys=len(base_hash), missing_keys=0,
                 extra_keys=0, value_mismatches=0, exact_match=True,
                 mismatch_examples="")]
    base_keys = set(base_hash)
    bad = False

    for stage in STAGES[1:]:
        state = normalize_state(load_state(chain[stage]))
        keys = {k for k,v in state.items() if is_teacher_key(k) and torch.is_tensor(v)}
        missing = sorted(base_keys - keys)
        extra = sorted(keys - base_keys)
        mismatched = [
            k for k in sorted(base_keys & keys)
            if tensor_digest(state[k]) != base_hash[k]
        ]
        exact = not missing and not extra and not mismatched
        bad |= not exact
        rows.append(dict(
            stage=stage, checked_keys=len(base_keys & keys),
            missing_keys=len(missing), extra_keys=len(extra),
            value_mismatches=len(mismatched), exact_match=exact,
            mismatch_examples=" | ".join((missing+extra+mismatched)[:12]),
        ))
        print(f"[teacher-check] {stage}: exact={exact}, mismatch={len(mismatched)}")
        del state
        gc.collect()

    sem.write_csv(out_csv, rows, [
        "stage","checked_keys","missing_keys","extra_keys",
        "value_mismatches","exact_match","mismatch_examples"
    ])
    if bad and not allow_mismatch:
        raise RuntimeError(
            f"Teacher-producing LAM tensors differ across CL stages. See {out_csv}"
        )


class FullAnchorCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, items):
        specs, samples = zip(*items)
        batch = self.base_collator(samples)
        batch["_trajectory_ids"] = [int(x["trajectory_id"]) for x in specs]
        batch["_trajectory_lengths"] = [int(x["trajectory_length"]) for x in specs]
        batch["_trajectory_steps"] = [int(x["trajectory_step"]) for x in specs]
        batch["_phase_fractions"] = [float(x["phase_fraction"]) for x in specs]
        batch["_phases"] = [str(x["phase"]) for x in specs]
        batch["_langs"] = [str(x["lang"]) for x in samples]
        return batch


def slim_full_batch(batch):
    out = {}
    for key in (*sem.VLM_KEYS, "primary_video", "embodiment_id"):
        v = batch[key]
        out[key] = v.detach().cpu().contiguous() if torch.is_tensor(v) else v
    for key in ("_trajectory_ids","_trajectory_lengths","_trajectory_steps",
                "_phase_fractions","_phases","_langs"):
        out[key] = list(batch[key])
    return out


def cache_ok(meta, args, task):
    return (
        meta.get("suite") == args.suite
        and int(meta.get("task_id",-1)) == int(task)
        and meta.get("split") == args.split
        and [float(x) for x in meta.get("trajectory_fractions",[])]
            == [float(x) for x in args.trajectory_fractions]
        and int(meta.get("max_trajectories_per_task",-999))
            == int(args.max_trajectories_per_task)
        and int(meta.get("batch_size",-1)) == int(args.batch_size)
        and bool(meta.get("contains_teacher_inputs",False))
        and int(meta.get("num_batches",0)) > 0
    )


def materialize_task(cfg, stats, collator, args, task, root):
    td = root / f"task_{task}"
    meta_path = td / "meta.json"
    if args.reuse_input_cache and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        paths = sorted(td.glob("batch_*.pt"))
        if cache_ok(meta,args,task) and len(paths)==int(meta["num_batches"]):
            print(f"[cache] task={task}: reuse {meta['num_samples']} anchors")
            return paths, meta

    if td.exists():
        shutil.rmtree(td)
    td.mkdir(parents=True)

    mixture = sem.get_vla_dataset(
        data_cfg=sem.task_data_cfg(cfg,args.suite,task),
        mode=args.split, balance_dataset_weights=False, seed=args.seed,
        framework_name=str(cfg.framework.name),
        dataset_statistics_override=stats,
    )
    for ds in getattr(mixture,"datasets",[]):
        tr = getattr(ds,"transforms",None)
        if tr is not None and hasattr(tr,"eval"):
            tr.eval()

    datasets = list(getattr(mixture,"datasets",[]))
    if len(datasets)!=1:
        raise RuntimeError(f"Expected one filtered dataset, got {len(datasets)}")
    single = datasets[0]

    specs, selected, warnings = sem.build_anchor_specs(
        single, args.trajectory_fractions, args.max_trajectories_per_task
    )
    for w in warnings:
        print("[WARN]",w)
    anchor_ds = sem.TrajectoryAnchorDataset(mixture,specs)

    kwargs = dict(
        dataset=anchor_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        collate_fn=FullAnchorCollator(collator), num_workers=args.num_workers,
        pin_memory=False, generator=torch.Generator().manual_seed(args.seed+task),
    )
    if args.num_workers>0:
        kwargs["worker_init_fn"]=sem.worker_init
        kwargs["persistent_workers"]=False
    loader = DataLoader(**kwargs)

    paths=[]; count=0; langs=set()
    for bi,batch in enumerate(loader):
        langs.update(str(x) for x in batch["_langs"])
        fixed=slim_full_batch(batch)
        path=td/f"batch_{bi:04d}.pt"
        torch.save(fixed,path); paths.append(path)
        count += int(fixed["input_ids"].shape[0])

    expected=len(selected)*len(args.trajectory_fractions)
    if count!=expected:
        raise RuntimeError(f"task={task}: anchors={count}, expected={expected}")
    if len(langs)!=1:
        raise RuntimeError(f"task={task}: expected one instruction, got {sorted(langs)}")

    meta=dict(
        suite=args.suite, task_id=int(task), split=args.split,
        trajectory_fractions=[float(x) for x in args.trajectory_fractions],
        max_trajectories_per_task=int(args.max_trajectories_per_task),
        batch_size=int(args.batch_size), num_batches=len(paths),
        available_trajectories=int(len(single.trajectory_ids)),
        num_trajectories=len(selected), num_samples=count,
        instruction=next(iter(langs)), contains_teacher_inputs=True,
    )
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    sem.write_anchor_manifest(td/"anchor_manifest.csv",args.suite,task,specs)
    print(f"[anchors] task={task}: trajectories={len(selected)}/{len(single.trajectory_ids)}, anchors={count}")
    return paths,meta


def load_stage_vlm_qformer(model, ckpt):
    # Validated VLM + learned-query loader from the earlier semantic analysis.
    sem.load_vlm_queries(model,ckpt)
    state=normalize_state(load_state(ckpt))
    prefix="policy_backend.vlm_to_lam."
    qstate={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    if not qstate:
        raise RuntimeError(f"No VLMToLAM keys in {ckpt}")
    result=model.policy_backend.vlm_to_lam.load_state_dict(qstate,strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"QFormer mismatch: {result}")
    del state,qstate
    gc.collect()


def get_hact(backend,fixed,device):
    hidden,attn,act_mask=sem.vlm_hidden_only(backend,fixed,device)
    _,_,hact=sem.select_features(hidden,attn,act_mask,int(backend.num_action_queries))
    del hidden,attn,act_mask
    return hact.detach()


def qforward(qformer,hact,device,dtype):
    ctx=(torch.autocast("cuda",dtype=dtype)
         if device.type=="cuda" else torch.autocast("cpu",enabled=False))
    with torch.inference_mode(),ctx:
        return qformer(hact).detach()


def teacher_forward(backend,fixed,device):
    with torch.inference_mode():
        return backend._run_lam_teacher(
            primary_video=fixed["primary_video"].to(device),
            embodiment_id=fixed["embodiment_id"].to(device),
        ).detach()


def rel_l2(cur,ref,eps=1e-8):
    cur,ref=cur.float(),ref.float()
    return float((torch.linalg.vector_norm(cur-ref)/
                  torch.linalg.vector_norm(ref).clamp_min(eps)).item())


def cos(cur,ref):
    return float(F.cosine_similarity(cur.float(),ref.float(),dim=-1,eps=1e-8).mean().item())


def mse(cur,ref):
    return float(F.mse_loss(cur.float(),ref.float()).item())


def save_ref(path,hbase,zbb,zstar,fixed):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(
        h_base=hbase.detach().cpu().to(torch.bfloat16),
        z_bb=zbb.detach().cpu().float(),
        z_star=zstar.detach().cpu().float(),
        trajectory_ids=list(fixed["_trajectory_ids"]),
        trajectory_lengths=list(fixed["_trajectory_lengths"]),
        trajectory_steps=list(fixed["_trajectory_steps"]),
        phase_fractions=list(fixed["_phase_fractions"]),
        phases=list(fixed["_phases"]),
        langs=list(fixed["_langs"]),
    ),path)


def row_meta(ref,i):
    return dict(
        trajectory_id=int(ref["trajectory_ids"][i]),
        trajectory_length=int(ref["trajectory_lengths"][i]),
        trajectory_step=int(ref["trajectory_steps"][i]),
        phase=str(ref["phases"][i]),
        phase_fraction=float(ref["phase_fractions"][i]),
        instruction=str(ref["langs"][i]),
    )


def make_rows(suite,task,stage,ref,routes,start):
    rows=[]
    zstar=ref["z_star"].float(); zbb=ref["z_bb"].float()
    route_cpu={k:v.detach().cpu().float() for k,v in routes.items()}
    for i in range(int(zstar.shape[0])):
        base_rel=rel_l2(zbb[i],zstar[i]); base_cos=cos(zbb[i],zstar[i])
        meta=row_meta(ref,i)
        for route in ROUTES:
            z=route_cpu[route][i]
            t_rel=rel_l2(z,zstar[i]); t_cos=cos(z,zstar[i])
            rows.append(dict(
                suite=suite,task_id=int(task),stage=stage,route=route,
                anchor_ordinal=int(start+i),**meta,
                teacher_rel_l2=t_rel,teacher_cosine=t_cos,teacher_mse=mse(z,zstar[i]),
                teacher_rel_l2_excess_vs_base=t_rel-base_rel,
                teacher_cosine_drop_vs_base=base_cos-t_cos,
                z_drift_rel_l2=rel_l2(z,zbb[i]),
                z_drift_cosine=cos(z,zbb[i]),
            ))
    return rows


def summarize(rows):
    # anchors -> trajectories
    g={}
    for r in rows:
        key=(r["stage"],r["route"],int(r["task_id"]),int(r["trajectory_id"]))
        g.setdefault(key,[]).append(r)
    traj=[]
    for key,rr in sorted(g.items()):
        stage,route,task,trid=key
        out=dict(stage=stage,route=route,task_id=task,trajectory_id=trid,n_anchors=len(rr))
        for m in METRICS: out[m]=statistics.fmean(float(x[m]) for x in rr)
        traj.append(out)

    # trajectories -> tasks
    g={}
    for r in traj:
        g.setdefault((r["stage"],r["route"],r["task_id"]),[]).append(r)
    tasks=[]
    for key,rr in sorted(g.items()):
        stage,route,task=key
        out=dict(stage=stage,route=route,task_id=task,n_trajectories=len(rr))
        for m in METRICS:
            vals=[float(x[m]) for x in rr]
            out[f"{m}_mean"]=statistics.fmean(vals)
            out[f"{m}_std"]=statistics.stdev(vals) if len(vals)>1 else 0.0
        tasks.append(out)

    # tasks -> macro
    stages=[]
    for stage in STAGES:
        for route in ROUTES:
            rr=[x for x in tasks if x["stage"]==stage and x["route"]==route]
            if not rr: continue
            out=dict(stage=stage,route=route,n_tasks=len(rr))
            for m in METRICS:
                vals=[float(x[f"{m}_mean"]) for x in rr]
                out[f"{m}_macro_mean"]=statistics.fmean(vals)
                out[f"{m}_std"]=statistics.stdev(vals) if len(vals)>1 else 0.0
            stages.append(out)
    return traj,tasks,stages


def write_matrix(path,tasks,task_ids,route,metric):
    lookup={(x["stage"],int(x["task_id"])):float(x[f"{metric}_mean"])
            for x in tasks if x["route"]==route}
    rows=[]
    for task in task_ids:
        row={"task":f"task_{task}"}
        for stage in STAGES: row[stage]=lookup.get((stage,int(task)),float("nan"))
        rows.append(row)
    macro={"task":"macro_mean"}
    for stage in STAGES:
        vals=[lookup[(stage,int(t))] for t in task_ids if (stage,int(t)) in lookup]
        macro[stage]=statistics.fmean(vals) if vals else float("nan")
    rows.append(macro)
    sem.write_csv(path,rows,["task",*STAGES])


def write_intervention(path,stage_rows):
    lookup={(x["stage"],x["route"]):x for x in stage_rows}
    rows=[]
    for stage in STAGES:
        row={"stage":stage}
        for route in ROUTES:
            item=lookup.get((stage,route))
            for m in METRICS:
                row[f"{route}_{m}"]=(
                    float(item[f"{m}_macro_mean"]) if item else float("nan")
                )
        rows.append(row)
    fields=["stage"]+[f"{r}_{m}" for r in ROUTES for m in METRICS]
    sem.write_csv(path,rows,fields)


def main():
    args=parse_args()
    sem.seed_all(args.seed)
    args.trajectory_fractions=sem.validate_fractions(args.trajectory_fractions)
    args.run_root=args.run_root.expanduser().resolve()
    args.output_dir=args.output_dir.expanduser().resolve()
    args.config_yaml=args.config_yaml.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)

    if args.split!="all": print(f"[WARN] formal protocol uses split=all, got {args.split}")
    if args.max_trajectories_per_task>0:
        print("[WARN] smoke/debug mode: max-trajectories-per-task is active")

    chain=sem.resolve_chain(args.run_root)
    cfg,stats=sem.load_base_cfg_stats(chain["Base"],args.config_yaml)

    print("="*88)
    print("LaWAM Latent-Action Forgetting -- 2x2 Module Intervention")
    print("="*88)
    print("suite:",args.suite," tasks:",args.task_ids)
    print("fractions:",args.trajectory_fractions," split:",args.split)
    for s in STAGES: print(f"{s:4s}: {chain[s]}")
    print("="*88)

    meta=dict(
        suite=args.suite,task_ids=args.task_ids,split=args.split,
        trajectory_fractions=args.trajectory_fractions,
        checkpoints={k:str(v) for k,v in chain.items()},
        routes=dict(
            base_ref="z_BB = Q_Base(H_Base)",
            qformer_only="z_Bk = Q_CLk(H_Base)",
            upstream_only="z_kB = Q_Base(H_CLk)",
            full="z_kk = Q_CLk(H_CLk)",
        ),
        teacher="fixed z* from Base LAM; vision_encoder+encoder+vq checked across stages",
    )
    (args.output_dir/"run_meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False))

    if not args.skip_teacher_stability_check:
        verify_teacher_stability(
            chain,args.output_dir/"teacher_stability_check.csv",
            args.allow_teacher_mismatch
        )

    # Fixed trajectory-balanced input cache.
    collator=sem.build_eval_collator(cfg)
    input_root=args.output_dir/"fixed_inputs"
    task_batches={}; coverage=[]
    for task in args.task_ids:
        paths,m=materialize_task(cfg,stats,collator,args,int(task),input_root)
        task_batches[int(task)]=paths
        coverage.append(dict(
            suite=args.suite,task_id=int(task),instruction=m["instruction"],
            available_trajectories=m["available_trajectories"],
            selected_trajectories=m["num_trajectories"],anchors=m["num_samples"]
        ))
    sem.write_csv(args.output_dir/"trajectory_coverage.csv",coverage,
                  ["suite","task_id","instruction","available_trajectories",
                   "selected_trajectories","anchors"])
    del collator; gc.collect()

    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("\n[model] loading Base LaWAM")
    model=sem.LaWAMFramework.from_pretrained(str(chain["Base"]))
    model.eval(); backend=model.policy_backend

    # Teacher z*: compute once with Base LAM.
    teacher_root=args.output_dir/"teacher_cache"
    if teacher_root.exists(): shutil.rmtree(teacher_root)
    teacher_root.mkdir(parents=True)
    backend.lam.to(device).eval()
    print("\n[teacher] computing fixed z*")
    for task in args.task_ids:
        count=0
        for bi,p in enumerate(task_batches[int(task)]):
            fixed=load_state(p)
            zstar=teacher_forward(backend,fixed,device).cpu().float()
            tp=teacher_root/f"task_{task}"/f"batch_{bi:04d}.pt"
            tp.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"z_star":zstar},tp)
            count+=int(zstar.shape[0])
            del fixed,zstar
            if device.type=="cuda": torch.cuda.empty_cache()
        print(f"[teacher] task={task}: {count} anchors")
    backend.lam.to("cpu"); gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()

    # Base H_B and z_BB.
    backend.vlm.to(device).eval()
    backend.vlm_to_lam.to(device).eval()
    base_qformer=copy.deepcopy(backend.vlm_to_lam).to(device).eval()
    for p in base_qformer.parameters(): p.requires_grad_(False)
    dtype=backend.model_cfg.vlm_dtype

    ref_root=args.output_dir/"base_reference"
    if ref_root.exists(): shutil.rmtree(ref_root)
    ref_root.mkdir(parents=True)
    print("\n[Base] computing H_B and z_BB")
    for task in args.task_ids:
        count=0
        for bi,p in enumerate(task_batches[int(task)]):
            fixed=load_state(p)
            zstar=load_state(teacher_root/f"task_{task}"/f"batch_{bi:04d}.pt")["z_star"]
            hbase=get_hact(backend,fixed,device)
            zbb=qforward(base_qformer,hbase,device,dtype)
            if tuple(zbb.shape)!=tuple(zstar.shape):
                raise RuntimeError(f"latent shape mismatch z_BB={zbb.shape}, z*={zstar.shape}")
            save_ref(ref_root/f"task_{task}"/f"batch_{bi:04d}.pt",hbase,zbb,zstar,fixed)
            count+=int(zbb.shape[0])
            del fixed,zstar,hbase,zbb
            if device.type=="cuda": torch.cuda.empty_cache()
        print(f"[Base] task={task}: {count} anchors")

    rows=[]

    # Base stage: all four intervention routes collapse to z_BB.
    for task in args.task_ids:
        ordinal=0
        for bi,p in enumerate(task_batches[int(task)]):
            ref=load_state(ref_root/f"task_{task}"/f"batch_{bi:04d}.pt")
            zbb=ref["z_bb"]
            routes={r:zbb for r in ROUTES}
            rows += make_rows(args.suite,task,"Base",ref,routes,ordinal)
            ordinal += int(zbb.shape[0])

    # CL stages.
    for stage in STAGES[1:]:
        print(f"\n[{stage}] loading VLM/queries/QFormer")
        load_stage_vlm_qformer(model,chain[stage])
        backend.vlm.eval(); backend.vlm_to_lam.eval()

        for task in args.task_ids:
            ordinal=0
            for bi,p in enumerate(task_batches[int(task)]):
                fixed=load_state(p)
                ref=load_state(ref_root/f"task_{task}"/f"batch_{bi:04d}.pt")
                hbase=ref["h_base"].to(device=device,dtype=torch.bfloat16)
                hk=get_hact(backend,fixed,device)

                zbk=qforward(backend.vlm_to_lam,hbase,device,dtype)
                zkb=qforward(base_qformer,hk,device,dtype)
                zkk=qforward(backend.vlm_to_lam,hk,device,dtype)
                zbb=ref["z_bb"]

                routes=dict(base_ref=zbb,qformer_only=zbk,upstream_only=zkb,full=zkk)
                rows += make_rows(args.suite,task,stage,ref,routes,ordinal)
                ordinal += int(zbb.shape[0])

                del fixed,ref,hbase,hk,zbk,zkb,zkk,zbb,routes
                if device.type=="cuda": torch.cuda.empty_cache()
            print(f"[{stage}] task={task}: {ordinal} anchors")

    # Detailed + hierarchical summaries.
    anchor_fields=[
        "suite","task_id","stage","route","anchor_ordinal","trajectory_id",
        "trajectory_length","trajectory_step","phase","phase_fraction","instruction",*METRICS
    ]
    sem.write_csv(args.output_dir/"latent_action_anchor_metrics.csv",rows,anchor_fields)
    traj,tasks,stages=summarize(rows)

    sem.write_csv(args.output_dir/"latent_action_trajectory_summary.csv",traj,
                  ["stage","route","task_id","trajectory_id","n_anchors",*METRICS])

    task_fields=["stage","route","task_id","n_trajectories"]
    for m in METRICS: task_fields += [f"{m}_mean",f"{m}_std"]
    sem.write_csv(args.output_dir/"latent_action_task_summary.csv",tasks,task_fields)

    stage_fields=["stage","route","n_tasks"]
    for m in METRICS: stage_fields += [f"{m}_macro_mean",f"{m}_std"]
    sem.write_csv(args.output_dir/"latent_action_stage_summary.csv",stages,stage_fields)

    write_intervention(args.output_dir/"latent_action_intervention_stage_summary.csv",stages)

    core=[
        ("full","teacher_rel_l2"),("full","teacher_cosine"),
        ("qformer_only","teacher_rel_l2"),("qformer_only","teacher_cosine"),
        ("upstream_only","teacher_rel_l2"),("upstream_only","teacher_cosine"),
        ("full","z_drift_rel_l2"),("full","z_drift_cosine"),
        ("full","teacher_rel_l2_excess_vs_base"),
        ("full","teacher_cosine_drop_vs_base"),
    ]
    for route,metric in core:
        write_matrix(args.output_dir/f"matrix_{route}_{metric}.csv",
                     tasks,args.task_ids,route,metric)

    print("\n"+"="*88)
    print("Latent-action intervention analysis complete")
    print("intervention:",args.output_dir/"latent_action_intervention_stage_summary.csv")
    print("core matrices:")
    for route,metric in core:
        print(" ",args.output_dir/f"matrix_{route}_{metric}.csv")
    print("="*88)


if __name__=="__main__":
    main()