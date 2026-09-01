#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import shutil
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

# Reuse the two analysis pipelines that have already been validated on the server:
#   semantic script -> trajectory-balanced anchors + fixed VLM inputs
#   latent-action script -> fixed z* teacher + stage VLM/QFormer loading
import analyze_libero_semantic_conditioning as sem
import analyze_libero_latent_action_forgetting as lat


STAGES = sem.STAGES

# Formal 2x2 world intervention at CL stage k:
#
#                      Base LaWM W_B           CL LaWM W_k
# teacher z*           W_B(h_t,z*)             W_k(h_t,z*)
# CL predicted z_k     W_B(h_t,z_k)            W_k(h_t,z_k)
#
ROUTES = ["base_ref", "wm_only", "upstream_only", "full"]

DECODER_PREFIX = "policy_backend.lam.decoder."

METRICS = (
    # Main future-vs-GT metrics.
    "future_rel_l2",
    "future_cosine",
    "future_mse",
    # LaWAM-paper-style initial-state comparison.
    "pred_init_cosine",
    "gt_init_cosine",
    "init_cosine_abs_error",
    # Relative to the pure Base world-model oracle route W_B(h_t,z*).
    "future_rel_l2_excess_vs_base_ref",
    "future_cosine_drop_vs_base_ref",
    # Relative to the actual Base policy world prediction W_B(h_t,z_BB).
    "future_rel_l2_excess_vs_base_policy",
    "future_cosine_drop_vs_base_policy",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "LaWAM World-Model / Dynamics Forgetting analysis with a fixed DINO/LAM "
            "reference and a 2x2 intervention over latent-action input {z*, z_CL} "
            "and world decoder {W_Base, W_CL}."
        )
    )
    p.add_argument("--suite", required=True, choices=["libero_goal", "libero_object"])
    p.add_argument(
        "--config-yaml",
        type=Path,
        default=Path("starVLA/config/training/train_libero.yaml"),
    )
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument(
        "--trajectory-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
    )
    p.add_argument(
        "--max-trajectories-per-task",
        type=int,
        default=0,
        help="0 = all trajectories; positive values are smoke/debug only.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--split", choices=["all", "train", "val"], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--reuse-input-cache", action="store_true")
    p.add_argument("--skip-teacher-stability-check", action="store_true")
    p.add_argument("--allow-teacher-mismatch", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Checkpoint sanity: LaWM decoder is the module that is allowed to change.
# -----------------------------------------------------------------------------

def tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(str(tuple(t.shape)).encode("utf-8"))
    if t.numel() > 0:
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def decoder_digest_map(ckpt: Path) -> dict[str, str]:
    state = lat.normalize_state(lat.load_state(ckpt))
    out = {
        k: tensor_digest(v)
        for k, v in state.items()
        if k.startswith(DECODER_PREFIX) and torch.is_tensor(v)
    }
    del state
    gc.collect()
    if not out:
        raise RuntimeError(f"No {DECODER_PREFIX} tensors found in {ckpt}")
    return out


def write_decoder_change_check(chain, out_csv: Path) -> None:
    base = decoder_digest_map(chain["Base"])
    base_keys = set(base)
    rows = [
        {
            "stage": "Base",
            "checked_keys": len(base),
            "missing_keys": 0,
            "extra_keys": 0,
            "value_mismatches_vs_base": 0,
            "exact_match_base": True,
            "mismatch_examples": "",
        }
    ]

    for stage in STAGES[1:]:
        current = decoder_digest_map(chain[stage])
        keys = set(current)
        missing = sorted(base_keys - keys)
        extra = sorted(keys - base_keys)
        mismatch = [
            k for k in sorted(base_keys & keys)
            if base[k] != current[k]
        ]
        rows.append(
            {
                "stage": stage,
                "checked_keys": len(base_keys & keys),
                "missing_keys": len(missing),
                "extra_keys": len(extra),
                "value_mismatches_vs_base": len(mismatch),
                "exact_match_base": not missing and not extra and not mismatch,
                "mismatch_examples": " | ".join((missing + extra + mismatch)[:12]),
            }
        )
        print(
            f"[decoder-check] {stage}: "
            f"changed_tensors={len(mismatch)} exact_base={not mismatch and not missing and not extra}"
        )

    sem.write_csv(
        out_csv,
        rows,
        [
            "stage",
            "checked_keys",
            "missing_keys",
            "extra_keys",
            "value_mismatches_vs_base",
            "exact_match_base",
            "mismatch_examples",
        ],
    )


def load_stage_decoder(model, ckpt: Path) -> None:
    state = lat.normalize_state(lat.load_state(ckpt))
    dstate = {
        k[len(DECODER_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(DECODER_PREFIX)
    }
    if not dstate:
        raise RuntimeError(f"No LaWM decoder state in {ckpt}")
    result = model.policy_backend.lam.decoder.load_state_dict(dstate, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"LaWM decoder mismatch for {ckpt}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    del state, dstate
    gc.collect()


# -----------------------------------------------------------------------------
# Exact LaWAM world-model computations
# -----------------------------------------------------------------------------

def lam_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def extract_fixed_world_targets(backend, fixed, device: torch.device):
    """Mirror LaWAM shared encoding for the frozen visual side.

    Returns:
      h_t   : DINO current latent visual feature
      h_gt  : DINO real future latent visual feature
      z*    : frozen LAM teacher latent action for the real transition
    """
    primary_video = fixed["primary_video"].to(device)
    embodiment_id = fixed["embodiment_id"].to(device)

    with torch.inference_mode(), lam_autocast(device):
        features = backend.lam.extract_vision_features(primary_video)
        if features is None or features.ndim != 4:
            raise RuntimeError(
                f"Expected LAM visual features [B,T,K,D], got "
                f"{None if features is None else tuple(features.shape)}"
            )
        h_t = features[:, 0, :, :]
        h_gt = features[:, -1, :, :]

    with torch.inference_mode():
        z_star = backend._run_lam_teacher(
            primary_video=primary_video,
            embodiment_id=embodiment_id,
        )

    return h_t.detach(), h_gt.detach(), z_star.detach()


def decode_future(decoder, h_t, z, device: torch.device):
    """Use the same single-query decoder convention as LaWAM."""
    if z.ndim != 3 or int(z.shape[1]) != 1:
        raise RuntimeError(f"World prediction expects z [B,1,D], got {tuple(z.shape)}")

    with torch.inference_mode(), lam_autocast(device):
        pred = decoder(h_t, z)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim == 4:
            pred = pred[:, 0, :, :] if int(pred.shape[1]) == 1 else pred[:, -1, :, :]

    if pred.ndim != 3:
        raise RuntimeError(f"Expected decoded future [B,K,D], got {tuple(pred.shape)}")
    return pred.detach()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def rel_l2(cur, ref, eps: float = 1e-8) -> float:
    cur, ref = cur.float(), ref.float()
    return float(
        (
            torch.linalg.vector_norm(cur - ref)
            / torch.linalg.vector_norm(ref).clamp_min(eps)
        ).item()
    )


def token_cos(cur, ref) -> float:
    return float(
        F.cosine_similarity(cur.float(), ref.float(), dim=-1, eps=1e-8)
        .mean()
        .item()
    )


def mse(cur, ref) -> float:
    return float(F.mse_loss(cur.float(), ref.float()).item())


def basic_future_metrics(pred, h_gt, h_t):
    pred_gt_cos = token_cos(pred, h_gt)
    pred_init_cos = token_cos(pred, h_t)
    gt_init_cos = token_cos(h_gt, h_t)
    return {
        "future_rel_l2": rel_l2(pred, h_gt),
        "future_cosine": pred_gt_cos,
        "future_mse": mse(pred, h_gt),
        "pred_init_cosine": pred_init_cos,
        "gt_init_cosine": gt_init_cos,
        "init_cosine_abs_error": abs(pred_init_cos - gt_init_cos),
    }


def row_meta(ref, i: int):
    return {
        "trajectory_id": int(ref["trajectory_ids"][i]),
        "trajectory_length": int(ref["trajectory_lengths"][i]),
        "trajectory_step": int(ref["trajectory_steps"][i]),
        "phase": str(ref["phases"][i]),
        "phase_fraction": float(ref["phase_fractions"][i]),
        "instruction": str(ref["langs"][i]),
    }


def make_rows(
    *,
    suite: str,
    task: int,
    stage: str,
    ref,
    route_predictions: dict[str, torch.Tensor],
    start_ordinal: int,
):
    h_t = ref["h_t"].float()
    h_gt = ref["h_gt"].float()
    base_ref_pred = ref["base_ref_pred"].float()
    base_policy_pred = ref["base_policy_pred"].float()

    base_ref_metrics = [
        basic_future_metrics(base_ref_pred[i], h_gt[i], h_t[i])
        for i in range(int(h_gt.shape[0]))
    ]
    base_policy_metrics = [
        basic_future_metrics(base_policy_pred[i], h_gt[i], h_t[i])
        for i in range(int(h_gt.shape[0]))
    ]

    routes_cpu = {
        k: v.detach().cpu().float()
        for k, v in route_predictions.items()
    }

    rows = []
    for i in range(int(h_gt.shape[0])):
        meta = row_meta(ref, i)
        for route in ROUTES:
            raw = basic_future_metrics(routes_cpu[route][i], h_gt[i], h_t[i])
            rows.append(
                {
                    "suite": suite,
                    "task_id": int(task),
                    "stage": stage,
                    "route": route,
                    "anchor_ordinal": int(start_ordinal + i),
                    **meta,
                    **raw,
                    "future_rel_l2_excess_vs_base_ref": (
                        raw["future_rel_l2"] - base_ref_metrics[i]["future_rel_l2"]
                    ),
                    "future_cosine_drop_vs_base_ref": (
                        base_ref_metrics[i]["future_cosine"] - raw["future_cosine"]
                    ),
                    "future_rel_l2_excess_vs_base_policy": (
                        raw["future_rel_l2"] - base_policy_metrics[i]["future_rel_l2"]
                    ),
                    "future_cosine_drop_vs_base_policy": (
                        base_policy_metrics[i]["future_cosine"] - raw["future_cosine"]
                    ),
                }
            )
    return rows


# -----------------------------------------------------------------------------
# Cache Base world references
# -----------------------------------------------------------------------------

def save_world_ref(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_world_reference_cache(
    *,
    backend,
    task_batches,
    task_ids,
    device,
    out_root: Path,
):
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    print("\n[world-ref] computing fixed h_t / h_GT / z* / Base-oracle prediction")
    backend.lam.to(device).eval()

    for task in task_ids:
        count = 0
        for bi, fixed_path in enumerate(task_batches[int(task)]):
            fixed = lat.load_state(fixed_path)
            h_t, h_gt, z_star = extract_fixed_world_targets(backend, fixed, device)
            base_ref_pred = decode_future(backend.lam.decoder, h_t, z_star, device)

            payload = {
                "h_t": h_t.cpu().to(torch.bfloat16).contiguous(),
                "h_gt": h_gt.cpu().to(torch.bfloat16).contiguous(),
                "z_star": z_star.cpu().float().contiguous(),
                "base_ref_pred": base_ref_pred.cpu().to(torch.bfloat16).contiguous(),
                "trajectory_ids": list(fixed["_trajectory_ids"]),
                "trajectory_lengths": list(fixed["_trajectory_lengths"]),
                "trajectory_steps": list(fixed["_trajectory_steps"]),
                "phase_fractions": list(fixed["_phase_fractions"]),
                "phases": list(fixed["_phases"]),
                "langs": list(fixed["_langs"]),
            }
            save_world_ref(
                out_root / f"task_{task}" / f"batch_{bi:04d}.pt",
                payload,
            )
            count += int(h_t.shape[0])
            del fixed, h_t, h_gt, z_star, base_ref_pred, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[world-ref] task={task}: {count} anchors")

    backend.lam.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def add_base_policy_predictions(
    *,
    backend,
    base_decoder,
    task_batches,
    task_ids,
    device,
    ref_root: Path,
):
    """Compute Base predicted z_BB and actual Base-policy world prediction W_B(h_t,z_BB)."""
    backend.vlm.to(device).eval()
    backend.vlm_to_lam.to(device).eval()
    base_decoder.to(device).eval()
    dtype = backend.model_cfg.vlm_dtype

    print("\n[Base] computing z_BB and W_B(h_t,z_BB)")
    for task in task_ids:
        count = 0
        for bi, fixed_path in enumerate(task_batches[int(task)]):
            fixed = lat.load_state(fixed_path)
            ref_path = ref_root / f"task_{task}" / f"batch_{bi:04d}.pt"
            ref = lat.load_state(ref_path)

            h_base = lat.get_hact(backend, fixed, device)
            z_bb = lat.qforward(
                backend.vlm_to_lam,
                h_base,
                device,
                dtype,
            )
            h_t = ref["h_t"].to(device=device, dtype=torch.bfloat16)
            base_policy_pred = decode_future(base_decoder, h_t, z_bb, device)

            ref["z_bb"] = z_bb.cpu().float().contiguous()
            ref["base_policy_pred"] = (
                base_policy_pred.cpu().to(torch.bfloat16).contiguous()
            )
            torch.save(ref, ref_path)

            count += int(z_bb.shape[0])
            del fixed, ref, h_base, z_bb, h_t, base_policy_pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[Base] task={task}: {count} anchors")


# -----------------------------------------------------------------------------
# Hierarchical statistics
# -----------------------------------------------------------------------------

def summarize(rows):
    # anchors -> trajectory
    groups = {}
    for row in rows:
        key = (
            row["stage"],
            row["route"],
            int(row["task_id"]),
            int(row["trajectory_id"]),
        )
        groups.setdefault(key, []).append(row)

    traj_rows = []
    for key, rr in sorted(groups.items()):
        stage, route, task, trajectory_id = key
        out = {
            "stage": stage,
            "route": route,
            "task_id": task,
            "trajectory_id": trajectory_id,
            "n_anchors": len(rr),
        }
        for metric in METRICS:
            out[metric] = statistics.fmean(float(x[metric]) for x in rr)
        traj_rows.append(out)

    # trajectory -> task
    groups = {}
    for row in traj_rows:
        groups.setdefault(
            (row["stage"], row["route"], int(row["task_id"])), []
        ).append(row)

    task_rows = []
    for key, rr in sorted(groups.items()):
        stage, route, task = key
        out = {
            "stage": stage,
            "route": route,
            "task_id": task,
            "n_trajectories": len(rr),
        }
        for metric in METRICS:
            vals = [float(x[metric]) for x in rr]
            out[f"{metric}_mean"] = statistics.fmean(vals)
            out[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        task_rows.append(out)

    # task -> suite macro
    stage_rows = []
    for stage in STAGES:
        for route in ROUTES:
            rr = [
                x for x in task_rows
                if x["stage"] == stage and x["route"] == route
            ]
            if not rr:
                continue
            out = {
                "stage": stage,
                "route": route,
                "n_tasks": len(rr),
            }
            for metric in METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in rr]
                out[f"{metric}_macro_mean"] = statistics.fmean(vals)
                out[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            stage_rows.append(out)

    return traj_rows, task_rows, stage_rows


def write_matrix(path, task_rows, task_ids, *, route: str, metric: str):
    lookup = {
        (x["stage"], int(x["task_id"])): float(x[f"{metric}_mean"])
        for x in task_rows
        if x["route"] == route
    }
    rows = []
    for task in task_ids:
        row = {"task": f"task_{task}"}
        for stage in STAGES:
            row[stage] = lookup.get((stage, int(task)), float("nan"))
        rows.append(row)

    macro = {"task": "macro_mean"}
    for stage in STAGES:
        vals = [
            lookup[(stage, int(task))]
            for task in task_ids
            if (stage, int(task)) in lookup
        ]
        macro[stage] = statistics.fmean(vals) if vals else float("nan")
    rows.append(macro)
    sem.write_csv(path, rows, ["task", *STAGES])


def write_intervention_summary(path, stage_rows):
    lookup = {
        (x["stage"], x["route"]): x
        for x in stage_rows
    }
    rows = []
    for stage in STAGES:
        row = {"stage": stage}
        for route in ROUTES:
            item = lookup.get((stage, route))
            for metric in METRICS:
                row[f"{route}_{metric}"] = (
                    float(item[f"{metric}_macro_mean"])
                    if item is not None
                    else float("nan")
                )
        rows.append(row)

    fields = ["stage"] + [
        f"{route}_{metric}"
        for route in ROUTES
        for metric in METRICS
    ]
    sem.write_csv(path, rows, fields)


def main():
    args = parse_args()
    sem.seed_all(args.seed)
    args.trajectory_fractions = sem.validate_fractions(args.trajectory_fractions)

    args.run_root = args.run_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config_yaml = args.config_yaml.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split != "all":
        print(f"[WARN] formal protocol uses split=all, got {args.split}")
    if args.max_trajectories_per_task > 0:
        print("[WARN] smoke/debug mode: max-trajectories-per-task is active")

    chain = sem.resolve_chain(args.run_root)
    cfg, stats = sem.load_base_cfg_stats(chain["Base"], args.config_yaml)

    print("=" * 92)
    print("LaWAM World-Model / Dynamics Forgetting -- 2x2 Intervention")
    print("=" * 92)
    print("suite                 :", args.suite)
    print("Base tasks            :", args.task_ids)
    print("fractions             :", args.trajectory_fractions)
    print("split                 :", args.split)
    print("max trajectories/task :", args.max_trajectories_per_task)
    for stage in STAGES:
        print(f"{stage:4s}: {chain[stage]}")
    print("=" * 92)

    run_meta = {
        "suite": args.suite,
        "task_ids": [int(x) for x in args.task_ids],
        "split": args.split,
        "trajectory_fractions": [float(x) for x in args.trajectory_fractions],
        "checkpoints": {k: str(v) for k, v in chain.items()},
        "routes": {
            "base_ref": "W_Base(h_t, z*)",
            "wm_only": "W_CLk(h_t, z*)",
            "upstream_only": "W_Base(h_t, z_CLk)",
            "full": "W_CLk(h_t, z_CLk)",
        },
        "references": {
            "h_t": "frozen LAM/DINO feature of current real observation",
            "h_gt": "frozen LAM/DINO feature of real future observation",
            "z_star": "fixed LAM teacher latent action inferred from the real transition",
            "base_policy": "W_Base(h_t, z_BB), used only as the actual Base-pipeline normalization",
        },
        "aggregation": (
            "25/50/75 anchors -> equal mean within trajectory -> "
            "equal mean across trajectories within task -> macro mean over Base tasks"
        ),
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # The same check already validated in Latent-Action Forgetting:
    # frozen vision encoder / IDM / VQ => fixed z* and fixed h_t/h_GT space.
    if not args.skip_teacher_stability_check:
        lat.verify_teacher_stability(
            chain,
            args.output_dir / "teacher_stability_check.csv",
            args.allow_teacher_mismatch,
        )

    # LaWM decoder is expected to be the changing world-model module.
    write_decoder_change_check(
        chain,
        args.output_dir / "decoder_change_check.csv",
    )

    # Materialize exactly the same trajectory-balanced inputs as the previous layer.
    collator = sem.build_eval_collator(cfg)
    input_root = args.output_dir / "fixed_inputs"
    task_batches = {}
    coverage = []

    for task in args.task_ids:
        paths, meta = lat.materialize_task(
            cfg,
            stats,
            collator,
            args,
            int(task),
            input_root,
        )
        task_batches[int(task)] = paths
        coverage.append(
            {
                "suite": args.suite,
                "task_id": int(task),
                "instruction": meta["instruction"],
                "available_trajectories": meta["available_trajectories"],
                "selected_trajectories": meta["num_trajectories"],
                "anchors": meta["num_samples"],
            }
        )

    sem.write_csv(
        args.output_dir / "trajectory_coverage.csv",
        coverage,
        [
            "suite",
            "task_id",
            "instruction",
            "available_trajectories",
            "selected_trajectories",
            "anchors",
        ],
    )
    del collator
    gc.collect()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("\n[model] loading Base LaWAM")
    model = sem.LaWAMFramework.from_pretrained(str(chain["Base"]))
    model.eval()
    backend = model.policy_backend

    # Preserve a permanent Base LaWM decoder for all controlled interventions.
    base_decoder = copy.deepcopy(backend.lam.decoder).eval()
    for p in base_decoder.parameters():
        p.requires_grad_(False)

    # 1) Fixed world space, fixed teacher, pure Base-world oracle prediction.
    ref_root = args.output_dir / "world_reference"
    build_world_reference_cache(
        backend=backend,
        task_batches=task_batches,
        task_ids=args.task_ids,
        device=device,
        out_root=ref_root,
    )

    # 2) Base predicted latent z_BB and actual Base policy future.
    add_base_policy_predictions(
        backend=backend,
        base_decoder=base_decoder,
        task_batches=task_batches,
        task_ids=args.task_ids,
        device=device,
        ref_root=ref_root,
    )

    # After the previous step, the VLM/QFormer and Base decoder remain on GPU.
    base_decoder.to(device).eval()
    backend.lam.decoder.to(device).eval()

    rows = []

    # Base stage:
    #   base_ref / wm_only  = W_B(h_t,z*)
    #   upstream / full     = W_B(h_t,z_BB), i.e. the actual Base policy world prediction.
    print("\n[Base] recording world intervention reference routes")
    for task in args.task_ids:
        ordinal = 0
        for bi, _ in enumerate(task_batches[int(task)]):
            ref = lat.load_state(
                ref_root / f"task_{task}" / f"batch_{bi:04d}.pt"
            )
            routes = {
                "base_ref": ref["base_ref_pred"],
                "wm_only": ref["base_ref_pred"],
                "upstream_only": ref["base_policy_pred"],
                "full": ref["base_policy_pred"],
            }
            rows += make_rows(
                suite=args.suite,
                task=int(task),
                stage="Base",
                ref=ref,
                route_predictions=routes,
                start_ordinal=ordinal,
            )
            ordinal += int(ref["h_gt"].shape[0])
        print(f"[Base] task={task}: {ordinal} anchors")

    # CL1-CL4:
    #   base_ref      = cached W_B(h_t,z*)
    #   wm_only       = W_k(h_t,z*)
    #   upstream_only = W_B(h_t,z_k)
    #   full          = W_k(h_t,z_k)
    dtype = backend.model_cfg.vlm_dtype
    for stage in STAGES[1:]:
        print(f"\n[{stage}] loading VLM/queries/QFormer + LaWM decoder")
        lat.load_stage_vlm_qformer(model, chain[stage])
        load_stage_decoder(model, chain[stage])

        backend.vlm.to(device).eval()
        backend.vlm_to_lam.to(device).eval()
        backend.lam.decoder.to(device).eval()

        for task in args.task_ids:
            ordinal = 0
            for bi, fixed_path in enumerate(task_batches[int(task)]):
                fixed = lat.load_state(fixed_path)
                ref = lat.load_state(
                    ref_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )

                # Current CL latent action z_k from the same correct old-task input.
                h_k = lat.get_hact(backend, fixed, device)
                z_k = lat.qforward(
                    backend.vlm_to_lam,
                    h_k,
                    device,
                    dtype,
                )

                h_t = ref["h_t"].to(device=device, dtype=torch.bfloat16)
                z_star = ref["z_star"].to(device=device)

                wm_only_pred = decode_future(
                    backend.lam.decoder,
                    h_t,
                    z_star,
                    device,
                )
                upstream_pred = decode_future(
                    base_decoder,
                    h_t,
                    z_k,
                    device,
                )
                full_pred = decode_future(
                    backend.lam.decoder,
                    h_t,
                    z_k,
                    device,
                )

                routes = {
                    "base_ref": ref["base_ref_pred"],
                    "wm_only": wm_only_pred,
                    "upstream_only": upstream_pred,
                    "full": full_pred,
                }
                rows += make_rows(
                    suite=args.suite,
                    task=int(task),
                    stage=stage,
                    ref=ref,
                    route_predictions=routes,
                    start_ordinal=ordinal,
                )
                ordinal += int(ref["h_gt"].shape[0])

                del (
                    fixed,
                    ref,
                    h_k,
                    z_k,
                    h_t,
                    z_star,
                    wm_only_pred,
                    upstream_pred,
                    full_pred,
                    routes,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            print(f"[{stage}] task={task}: {ordinal} anchors")

    # Detailed + hierarchical summaries.
    anchor_fields = [
        "suite",
        "task_id",
        "stage",
        "route",
        "anchor_ordinal",
        "trajectory_id",
        "trajectory_length",
        "trajectory_step",
        "phase",
        "phase_fraction",
        "instruction",
        *METRICS,
    ]
    sem.write_csv(
        args.output_dir / "world_anchor_metrics.csv",
        rows,
        anchor_fields,
    )

    traj_rows, task_rows, stage_rows = summarize(rows)

    sem.write_csv(
        args.output_dir / "world_trajectory_summary.csv",
        traj_rows,
        ["stage", "route", "task_id", "trajectory_id", "n_anchors", *METRICS],
    )

    task_fields = ["stage", "route", "task_id", "n_trajectories"]
    for metric in METRICS:
        task_fields += [f"{metric}_mean", f"{metric}_std"]
    sem.write_csv(
        args.output_dir / "world_task_summary.csv",
        task_rows,
        task_fields,
    )

    stage_fields = ["stage", "route", "n_tasks"]
    for metric in METRICS:
        stage_fields += [f"{metric}_macro_mean", f"{metric}_std"]
    sem.write_csv(
        args.output_dir / "world_stage_summary.csv",
        stage_rows,
        stage_fields,
    )

    write_intervention_summary(
        args.output_dir / "world_intervention_stage_summary.csv",
        stage_rows,
    )

    # Core matrices for the paper/mechanism analysis.
    core = [
        # Full actual CL pipeline.
        ("full", "future_rel_l2"),
        ("full", "future_cosine"),
        # Pure world-model decoder forgetting.
        ("wm_only", "future_rel_l2"),
        ("wm_only", "future_cosine"),
        # Propagation from latent-action forgetting into future prediction.
        ("upstream_only", "future_rel_l2"),
        ("upstream_only", "future_cosine"),
        # LaWAM-style predicted-future vs initial-state diagnostic.
        ("full", "pred_init_cosine"),
        ("wm_only", "pred_init_cosine"),
        # Base-normalized views.
        ("wm_only", "future_rel_l2_excess_vs_base_ref"),
        ("wm_only", "future_cosine_drop_vs_base_ref"),
        ("full", "future_rel_l2_excess_vs_base_policy"),
        ("full", "future_cosine_drop_vs_base_policy"),
    ]
    for route, metric in core:
        write_matrix(
            args.output_dir / f"matrix_{route}_{metric}.csv",
            task_rows,
            args.task_ids,
            route=route,
            metric=metric,
        )

    print("\n" + "=" * 92)
    print("World-model / dynamics forgetting analysis complete")
    print("=" * 92)
    print("teacher check :", args.output_dir / "teacher_stability_check.csv")
    print("decoder check :", args.output_dir / "decoder_change_check.csv")
    print("intervention  :", args.output_dir / "world_intervention_stage_summary.csv")
    print("core matrices :")
    for route, metric in core:
        print(" ", args.output_dir / f"matrix_{route}_{metric}.csv")
    print("=" * 92)


if __name__ == "__main__":
    main()
