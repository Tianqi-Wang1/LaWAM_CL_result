#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import faulthandler
import gc
import hashlib
import json
import math
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the already validated trajectory-balanced LaWAM analysis helpers.
import analyze_libero_semantic_conditioning as sem
import analyze_libero_latent_action_forgetting as lat

from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid


faulthandler.enable(all_threads=True)

STAGES = sem.STAGES

# Final action-layer intervention protocol:
#
# route            VLM condition    future condition    Flow
# --------------------------------------------------------------
# base_ref         Base             Base                Base
# vlm_only         CLk              Base                Base
# world_only       Base             CLk                 Base
# upstream_joint   CLk              CLk                 Base
# flow_only        Base             Base                CLk
# full             CLk              CLk                 CLk
#
ROUTES = [
    "base_ref",
    "vlm_only",
    "world_only",
    "upstream_joint",
    "flow_only",
    "full",
]

FLOW_PREFIX = "policy_backend.flow."
DECODER_PREFIX = "policy_backend.lam.decoder."

FIXED_BATCH_KEYS = (
    *sem.VLM_KEYS,
    "primary_video",
    "state",
    "state_mask",
    "embodiment_id",
    "action_hz",
    "actions",
    "actions_mask",
)

METRICS = (
    "flow_mse",
    "flow_rmse",
    "flow_mse_excess_vs_base",
    "action_mse",
    "action_rmse",
    "action_mse_excess_vs_base",
)

COUPLING_METRICS = (
    "vlm_excess_flow_mse",
    "world_excess_flow_mse",
    "upstream_joint_excess_flow_mse",
    "coupling_interaction_flow_mse",
    "vlm_excess_action_mse",
    "world_excess_action_mse",
    "upstream_joint_excess_action_mse",
    "coupling_interaction_action_mse",
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "LaWAM Action Generation / Conditioning-Coupling Forgetting analysis. "
            "Primary metric: deterministic fixed-(noise,time) flow-matching MSE. "
            "Six controlled routes separate direct VLM, world, joint-upstream, "
            "and Flow-head forgetting."
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
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--split", choices=["all", "train", "val"], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument(
        "--num-flow-probes",
        type=int,
        default=4,
        help=(
            "Number of fixed (epsilon,tau) probes per anchor. "
            "All stages/routes share exactly the same probes."
        ),
    )

    p.add_argument(
        "--enable-sampled-action",
        action="store_true",
        help=(
            "Also run deterministic fixed-initial-noise action sampling as an auxiliary metric. "
            "This is much slower than the primary flow-MSE probe."
        ),
    )
    p.add_argument(
        "--sampled-action-steps",
        type=int,
        default=0,
        help="0 = use checkpoint Flow config num_inference_steps (LIBERO default 10).",
    )

    p.add_argument("--reuse-input-cache", action="store_true")
    p.add_argument(
        "--keep-stage-condition-cache",
        action="store_true",
        help="Keep temporary CL-stage H_VLM/future-condition caches after each stage.",
    )
    return p.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================

def load_cache(path: Path):
    """Analysis caches are deliberately loaded without mmap."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(str(tuple(t.shape)).encode("utf-8"))
    if t.numel():
        h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def module_digest_map(ckpt: Path, prefix: str) -> dict[str, str]:
    state = lat.normalize_state(lat.load_state(ckpt))
    out = {
        k: tensor_digest(v)
        for k, v in state.items()
        if k.startswith(prefix) and torch.is_tensor(v)
    }
    del state
    gc.collect()
    if not out:
        raise RuntimeError(f"No checkpoint tensors found under {prefix!r}: {ckpt}")
    return out


def write_module_change_check(chain, prefix: str, out_csv: Path, label: str) -> None:
    base = module_digest_map(chain["Base"], prefix)
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
        cur = module_digest_map(chain[stage], prefix)
        keys = set(cur)
        missing = sorted(base_keys - keys)
        extra = sorted(keys - base_keys)
        mismatch = [k for k in sorted(base_keys & keys) if base[k] != cur[k]]
        exact = not missing and not extra and not mismatch
        rows.append(
            {
                "stage": stage,
                "checked_keys": len(base_keys & keys),
                "missing_keys": len(missing),
                "extra_keys": len(extra),
                "value_mismatches_vs_base": len(mismatch),
                "exact_match_base": exact,
                "mismatch_examples": " | ".join((missing + extra + mismatch)[:12]),
            }
        )
        print(
            f"[{label}-check] {stage}: changed_tensors={len(mismatch)} "
            f"exact_base={exact}"
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


def load_stage_flow(model, ckpt: Path) -> None:
    state = lat.normalize_state(lat.load_state(ckpt))
    fstate = {
        k[len(FLOW_PREFIX):]: v
        for k, v in state.items()
        if k.startswith(FLOW_PREFIX)
    }
    if not fstate:
        raise RuntimeError(f"No Flow state in {ckpt}")
    result = model.policy_backend.flow.load_state_dict(fstate, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Flow mismatch for {ckpt}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    del state, fstate
    gc.collect()


def lam_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def decode_future(decoder, h_t, z, device: torch.device):
    if z.ndim != 3 or int(z.shape[1]) != 1:
        raise RuntimeError(f"Expected latent action [B,1,D], got {tuple(z.shape)}")
    with torch.inference_mode(), lam_autocast(device):
        pred = decoder(h_t, z)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim == 4:
            pred = pred[:, 0] if int(pred.shape[1]) == 1 else pred[:, -1]
    if pred.ndim != 3:
        raise RuntimeError(f"Expected future feature [B,K,D], got {tuple(pred.shape)}")
    return pred.detach()


# =============================================================================
# Exact action-analysis cache
# =============================================================================

class ActionAnchorCollator:
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


def slim_action_batch(batch):
    out = {}
    for key in FIXED_BATCH_KEYS:
        value = batch[key]
        out[key] = (
            value.detach().cpu().contiguous()
            if torch.is_tensor(value)
            else value
        )
    for key in (
        "_trajectory_ids",
        "_trajectory_lengths",
        "_trajectory_steps",
        "_phase_fractions",
        "_phases",
        "_langs",
    ):
        out[key] = list(batch[key])
    return out


def action_cache_ok(meta, args, task):
    return (
        meta.get("suite") == args.suite
        and int(meta.get("task_id", -1)) == int(task)
        and meta.get("split") == args.split
        and [float(x) for x in meta.get("trajectory_fractions", [])]
        == [float(x) for x in args.trajectory_fractions]
        and int(meta.get("max_trajectories_per_task", -999))
        == int(args.max_trajectories_per_task)
        and int(meta.get("batch_size", -1)) == int(args.batch_size)
        and bool(meta.get("contains_action_inputs", False))
        and int(meta.get("num_batches", 0)) > 0
    )


def materialize_action_task(cfg, stats, collator, args, task, root):
    td = root / f"task_{task}"
    meta_path = td / "meta.json"

    if args.reuse_input_cache and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        paths = sorted(td.glob("batch_*.pt"))
        if action_cache_ok(meta, args, task) and len(paths) == int(meta["num_batches"]):
            print(f"[cache] task={task}: reuse {meta['num_samples']} action anchors")
            return paths, meta

    if td.exists():
        shutil.rmtree(td)
    td.mkdir(parents=True, exist_ok=True)

    mixture = sem.get_vla_dataset(
        data_cfg=sem.task_data_cfg(cfg, args.suite, task),
        mode=args.split,
        balance_dataset_weights=False,
        seed=args.seed,
        framework_name=str(cfg.framework.name),
        dataset_statistics_override=stats,
    )
    for ds in getattr(mixture, "datasets", []):
        transforms = getattr(ds, "transforms", None)
        if transforms is not None and hasattr(transforms, "eval"):
            transforms.eval()

    datasets = list(getattr(mixture, "datasets", []))
    if len(datasets) != 1:
        raise RuntimeError(f"Expected one filtered LIBERO dataset, got {len(datasets)}")
    single = datasets[0]

    specs, selected, warnings = sem.build_anchor_specs(
        single,
        args.trajectory_fractions,
        args.max_trajectories_per_task,
    )
    for warning in warnings:
        print("[WARN]", warning)
    if not specs:
        raise RuntimeError(f"No trajectory anchors for task={task}")

    anchor_ds = sem.TrajectoryAnchorDataset(mixture, specs)
    kwargs = dict(
        dataset=anchor_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=ActionAnchorCollator(collator),
        num_workers=args.num_workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(args.seed + task),
    )
    if args.num_workers > 0:
        kwargs["worker_init_fn"] = sem.worker_init
        kwargs["persistent_workers"] = False

    loader = DataLoader(**kwargs)
    paths = []
    count = 0
    langs = set()
    for bi, batch in enumerate(loader):
        langs.update(str(x) for x in batch["_langs"])
        fixed = slim_action_batch(batch)
        path = td / f"batch_{bi:04d}.pt"
        torch.save(fixed, path)
        paths.append(path)
        count += int(fixed["input_ids"].shape[0])

    expected = len(selected) * len(args.trajectory_fractions)
    if count != expected:
        raise RuntimeError(f"task={task}: anchors={count}, expected={expected}")
    if len(langs) != 1:
        raise RuntimeError(f"task={task}: expected one instruction, got {sorted(langs)}")

    meta = {
        "suite": args.suite,
        "task_id": int(task),
        "split": args.split,
        "trajectory_fractions": [float(x) for x in args.trajectory_fractions],
        "max_trajectories_per_task": int(args.max_trajectories_per_task),
        "batch_size": int(args.batch_size),
        "num_batches": len(paths),
        "available_trajectories": int(len(single.trajectory_ids)),
        "num_trajectories": len(selected),
        "num_samples": count,
        "instruction": next(iter(langs)),
        "contains_action_inputs": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    sem.write_anchor_manifest(td / "anchor_manifest.csv", args.suite, task, specs)
    print(
        f"[anchors] task={task}: trajectories={len(selected)}/{len(single.trajectory_ids)}, "
        f"anchors={count}"
    )
    return paths, meta


# =============================================================================
# Base / CL action conditions
# =============================================================================

def get_full_vlm_and_hact(backend, fixed, device):
    hidden, attention_mask, act_mask = sem.vlm_hidden_only(backend, fixed, device)
    _, _, hact = sem.select_features(
        hidden,
        attention_mask,
        act_mask,
        int(backend.num_action_queries),
    )
    return hidden.detach(), hact.detach()


def extract_h_t(backend, fixed, device):
    primary_video = fixed["primary_video"].to(device)
    with torch.inference_mode(), lam_autocast(device):
        features = backend.lam.extract_vision_features(primary_video)
        if features is None or features.ndim != 4:
            raise RuntimeError(
                f"Expected LAM visual features [B,T,K,D], got "
                f"{None if features is None else tuple(features.shape)}"
            )
        h_t = features[:, 0]
    return h_t.detach()


def save_condition(path: Path, h_t, h_vlm, h_future):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "h_t": h_t.detach().cpu().to(torch.bfloat16).contiguous(),
            "h_vlm": h_vlm.detach().cpu().to(torch.bfloat16).contiguous(),
            "h_future": h_future.detach().cpu().to(torch.bfloat16).contiguous(),
        },
        path,
    )


def build_base_condition_cache(
    backend,
    task_batches,
    task_ids,
    device,
    out_root: Path,
):
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    backend.lam.to(device).eval()
    backend.vlm.to(device).eval()
    backend.vlm_to_lam.to(device).eval()
    dtype = backend.model_cfg.vlm_dtype

    print("\n[Base] computing H_VLM^B, h_t, z_BB, and future_B")
    for task in task_ids:
        count = 0
        for bi, fixed_path in enumerate(task_batches[int(task)]):
            fixed = load_cache(fixed_path)
            h_t = extract_h_t(backend, fixed, device)
            h_vlm, hact = get_full_vlm_and_hact(backend, fixed, device)
            z_bb = lat.qforward(
                backend.vlm_to_lam,
                hact,
                device,
                dtype,
            )
            h_future = decode_future(
                backend.lam.decoder,
                h_t,
                z_bb,
                device,
            )
            save_condition(
                out_root / f"task_{task}" / f"batch_{bi:04d}.pt",
                h_t,
                h_vlm,
                h_future,
            )
            count += int(h_t.shape[0])
            del fixed, h_t, h_vlm, hact, z_bb, h_future
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[Base] task={task}: {count} condition anchors")

    backend.lam.to("cpu")
    backend.vlm.to("cpu")
    backend.vlm_to_lam.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_stage_condition_cache(
    model,
    ckpt: Path,
    task_batches,
    task_ids,
    base_root: Path,
    device,
    out_root: Path,
):
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    lat.load_stage_vlm_qformer(model, ckpt)
    load_stage_decoder(model, ckpt)
    backend = model.policy_backend

    backend.vlm.to(device).eval()
    backend.vlm_to_lam.to(device).eval()
    backend.lam.decoder.to(device).eval()
    dtype = backend.model_cfg.vlm_dtype

    for task in task_ids:
        count = 0
        for bi, fixed_path in enumerate(task_batches[int(task)]):
            fixed = load_cache(fixed_path)
            base = load_cache(base_root / f"task_{task}" / f"batch_{bi:04d}.pt")
            h_t = base["h_t"].to(device=device, dtype=torch.bfloat16)

            h_vlm, hact = get_full_vlm_and_hact(backend, fixed, device)
            z_k = lat.qforward(
                backend.vlm_to_lam,
                hact,
                device,
                dtype,
            )
            h_future = decode_future(
                backend.lam.decoder,
                h_t,
                z_k,
                device,
            )
            save_condition(
                out_root / f"task_{task}" / f"batch_{bi:04d}.pt",
                h_t,
                h_vlm,
                h_future,
            )
            count += int(h_t.shape[0])

            del fixed, base, h_t, h_vlm, hact, z_k, h_future
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"[conditions] task={task}: {count} CL anchors")

    backend.vlm.to("cpu")
    backend.vlm_to_lam.to("cpu")
    backend.lam.decoder.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# Fixed flow probes
# =============================================================================

def make_fixed_probes(flow, fixed, num_probes: int, seed: int, device):
    """Generate exact LaWAM noise/time probes once, then reuse for every stage/route."""
    actions = fixed["actions"]
    bsz, horizon, action_dim = map(int, actions.shape)
    model_dtype = flow._compute_dtype()

    noises = []
    times = []

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        for _ in range(int(num_probes)):
            noise = flow.sample_noise(
                (bsz, horizon, action_dim),
                device=device,
                dtype=model_dtype,
            )
            if flow.config.token_independent_noise:
                time = flow.sample_time(
                    bsz * horizon,
                    device,
                    model_dtype,
                ).view(bsz, horizon, 1)
            else:
                time = flow.sample_time(
                    bsz,
                    device,
                    model_dtype,
                )[:, None, None]
            noises.append(noise.detach().cpu().float())
            times.append(time.detach().cpu().float())

    return {
        "noise": torch.stack(noises, dim=0),  # [P,B,T,K]
        "time": torch.stack(times, dim=0),    # [P,B,1,1] or [P,B,T,1]
    }


def repeat_probe_batch(x: torch.Tensor, probes: int) -> torch.Tensor:
    return x.repeat(probes, *([1] * (x.ndim - 1)))


def deterministic_flow_per_sample_mse(
    flow,
    *,
    h_t,
    h_future,
    h_vlm,
    state,
    actions,
    action_hz,
    embodiment_id,
    state_mask,
    actions_mask,
    attention_mask,
    noise,
    time,
):
    """Mirror ConditionalFlowMatchingHead.forward with fixed epsilon/tau.

    Returns per-sample masked velocity MSE instead of one batch-reduced scalar.
    """
    model_dtype = flow._compute_dtype()

    h_t = flow._cast_if_needed(h_t, model_dtype)
    h_future = flow._cast_if_needed(h_future, model_dtype)
    h_vlm = flow._cast_if_needed(h_vlm, model_dtype)
    actions = flow._cast_if_needed(actions, model_dtype)
    noise = flow._cast_if_needed(noise, model_dtype)
    time = flow._cast_if_needed(time, model_dtype)

    device = actions.device
    batch_size = int(actions.shape[0])

    actions_mask_f = actions_mask.to(device=device, dtype=model_dtype)
    action_hz_f = action_hz.to(device=device, dtype=torch.float32)
    data_token_valid = actions_mask.to(device=device, dtype=torch.bool).any(dim=-1)

    x_t = (1 - time) * noise + time * actions
    velocity_target = actions - noise

    t_discretized = (
        time.squeeze(-1) * int(flow.config.num_timestep_buckets)
    ).long()
    t_discretized = torch.clamp(
        t_discretized,
        0,
        int(flow.config.num_timestep_buckets) - 1,
    )
    if not flow.config.token_independent_noise:
        t_discretized = t_discretized[:, 0]

    t_grid, hz_token_valid, _ = build_time_grid(
        horizon_sec=float(flow.config.horizon_sec),
        hz=action_hz_f,
        seq_len=int(actions.shape[1]),
    )
    expected_total = int(hz_token_valid.sum().item())
    actual_total = int(data_token_valid.sum().item())
    if actual_total != expected_total:
        raise RuntimeError(
            "Action mask/time-grid mismatch in deterministic flow probe: "
            f"data_total={actual_total}, hz_total={expected_total}"
        )
    token_valid = data_token_valid

    noisy_trajectory_emb = flow.action_encoder(x_t, embodiment_id)
    time_emb = flow.time_encoder(t_grid).to(dtype=noisy_trajectory_emb.dtype)
    action_time_emb = (
        time_emb
        * token_valid.unsqueeze(-1).to(dtype=noisy_trajectory_emb.dtype)
    )
    noisy_trajectory_emb = (
        noisy_trajectory_emb
        * token_valid.unsqueeze(-1).to(dtype=noisy_trajectory_emb.dtype)
    )

    cond_state = flow._prepare_state_condition(
        state=state,
        state_mask=state_mask,
        embodiment_id=embodiment_id,
        model_dtype=model_dtype,
    )
    cond_vlm = flow.enc_vlm(h_vlm)
    cond_future = h_future
    encoder_hidden_states = torch.cat((h_t, cond_future, cond_vlm), dim=1)

    future_tokens, future_token_valid = flow._expand_future_tokens(
        batch_size=batch_size,
        device=device,
        dtype=noisy_trajectory_emb.dtype,
    )
    future_token_count = 0 if future_tokens is None else int(future_tokens.shape[1])

    hidden_positional_embeddings = None
    if flow.config.use_action_positional_embeddings:
        hidden_positional_embeddings = flow._build_hidden_positional_embeddings(
            action_time_emb=action_time_emb,
            batch_size=batch_size,
            device=device,
            dtype=noisy_trajectory_emb.dtype,
            has_state_token=bool(flow.config.use_state),
            future_token_count=future_token_count,
        )

    dit_timestep = t_discretized
    if flow.config.token_independent_noise:
        dit_timestep = flow._build_hidden_timesteps(
            action_timesteps=t_discretized,
            token_valid=token_valid,
            has_state_token=bool(flow.config.use_state),
            future_token_count=future_token_count,
        )

    if flow.config.use_state:
        state_token_valid = torch.ones(
            (batch_size, 1),
            dtype=torch.bool,
            device=device,
        )
        if future_tokens is not None and future_token_valid is not None:
            hidden_states = torch.cat(
                (cond_state, future_tokens, noisy_trajectory_emb),
                dim=1,
            )
            hidden_attention_mask = torch.cat(
                [state_token_valid, future_token_valid, token_valid],
                dim=1,
            )
        else:
            hidden_states = torch.cat((cond_state, noisy_trajectory_emb), dim=1)
            hidden_attention_mask = torch.cat(
                [state_token_valid, token_valid],
                dim=1,
            )
    else:
        if future_tokens is not None and future_token_valid is not None:
            hidden_states = torch.cat(
                (future_tokens, noisy_trajectory_emb),
                dim=1,
            )
            hidden_attention_mask = torch.cat(
                [future_token_valid, token_valid],
                dim=1,
            )
        else:
            hidden_states = noisy_trajectory_emb
            hidden_attention_mask = token_valid

    num_vision = int(h_t.shape[1] + cond_future.shape[1])
    num_vlm = int(cond_vlm.shape[1])

    if attention_mask is not None:
        vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
        vision_mask_bool = torch.ones(
            batch_size,
            num_vision,
            dtype=torch.bool,
            device=device,
        )
        encoder_attention_mask = torch.cat(
            [vision_mask_bool, vlm_mask_bool],
            dim=1,
        )
    else:
        encoder_attention_mask = None

    if flow.config.use_alternate_vldit:
        image_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    num_vision,
                    dtype=torch.bool,
                    device=device,
                ),
                torch.zeros(
                    batch_size,
                    num_vlm,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )
        vlm_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    num_vision,
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_size,
                    num_vlm,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        )
        dit_output = flow.DiT(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=dit_timestep,
            hidden_attention_mask=hidden_attention_mask,
            image_mask=image_mask,
            vlm_mask=vlm_mask,
            encoder_attention_mask=encoder_attention_mask,
            hidden_positional_embeddings=hidden_positional_embeddings,
        )
    else:
        dit_output = flow.DiT(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=dit_timestep,
            hidden_attention_mask=hidden_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            hidden_positional_embeddings=hidden_positional_embeddings,
        )

    pred_velocity_all = flow.action_decoder(dit_output, embodiment_id)
    pred_velocity = pred_velocity_all[:, -actions.shape[1]:, :]

    loss_elem = F.mse_loss(
        pred_velocity.float(),
        velocity_target.float(),
        reduction="none",
    )
    valid = actions_mask_f.float()
    robot_valid = (
        embodiment_id.to(device=device, dtype=torch.long) != 0
    ).float()
    valid = valid * robot_valid.view(-1, 1, 1)

    numerator = (loss_elem * valid).sum(dim=(1, 2))
    denominator = valid.sum(dim=(1, 2)).clamp_min(1.0)
    return numerator / denominator


def masked_action_mse_per_sample(pred, target, mask, embodiment_id):
    pred = pred.float()
    target = target.float()
    mask_f = mask.to(device=pred.device, dtype=torch.float32)
    robot_valid = (
        embodiment_id.to(device=pred.device, dtype=torch.long) != 0
    ).float()
    mask_f = mask_f * robot_valid.view(-1, 1, 1)
    err = (pred - target).square()
    return (
        (err * mask_f).sum(dim=(1, 2))
        / mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    )


def deterministic_sample_actions(
    flow,
    *,
    h_t,
    h_future,
    h_vlm,
    state,
    state_mask,
    action_hz,
    embodiment_id,
    attention_mask,
    initial_noise,
    num_inference_steps: int,
):
    """Use official sample_actions_cfg but replace its one initial noise draw."""
    expected_shape = tuple(initial_noise.shape)

    had_instance_attr = "sample_noise" in flow.__dict__
    old_instance_attr = flow.__dict__.get("sample_noise", None)

    def fixed_noise(shape, device, dtype):
        if tuple(shape) != expected_shape:
            raise RuntimeError(
                f"Fixed sample noise shape mismatch: requested={tuple(shape)}, "
                f"cached={expected_shape}"
            )
        return initial_noise.to(device=device, dtype=dtype).clone()

    flow.sample_noise = fixed_noise
    try:
        pred = flow.sample_actions_cfg(
            h_t=h_t,
            h_t1_star=h_future,
            h_vlm=h_vlm,
            state=state,
            state_mask=state_mask,
            action_hz=action_hz,
            embodiment_id=embodiment_id,
            cfg_scale=1.0,
            num_inference_steps=int(num_inference_steps),
            attention_mask=attention_mask,
            return_padded=True,
        )
    finally:
        if had_instance_attr:
            flow.sample_noise = old_instance_attr
        else:
            delattr(flow, "sample_noise")

    return pred.detach()


def evaluate_route(
    flow,
    *,
    fixed,
    h_t,
    h_vlm,
    h_future,
    probes,
    device,
    enable_sampled_action: bool,
    sampled_action_steps: int,
):
    flow.to(device).eval()
    probes_n = int(probes["noise"].shape[0])
    bsz = int(fixed["actions"].shape[0])

    actions = fixed["actions"].to(device)
    actions_mask = fixed["actions_mask"].to(device)
    state = fixed["state"].to(device)
    state_mask = fixed["state_mask"].to(device)
    action_hz = fixed["action_hz"].to(device)
    embodiment_id = fixed["embodiment_id"].to(device)
    attention_mask = fixed["attention_mask"].to(device)

    h_t = h_t.to(device)
    h_vlm = h_vlm.to(device)
    h_future = h_future.to(device)

    noise = probes["noise"].to(device)
    time = probes["time"].to(device)

    noise_flat = noise.reshape(
        probes_n * bsz,
        *noise.shape[2:],
    )
    time_flat = time.reshape(
        probes_n * bsz,
        *time.shape[2:],
    )

    with torch.inference_mode():
        per_probe = deterministic_flow_per_sample_mse(
            flow,
            h_t=repeat_probe_batch(h_t, probes_n),
            h_future=repeat_probe_batch(h_future, probes_n),
            h_vlm=repeat_probe_batch(h_vlm, probes_n),
            state=repeat_probe_batch(state, probes_n),
            actions=repeat_probe_batch(actions, probes_n),
            action_hz=repeat_probe_batch(action_hz, probes_n),
            embodiment_id=repeat_probe_batch(embodiment_id, probes_n),
            state_mask=repeat_probe_batch(state_mask, probes_n),
            actions_mask=repeat_probe_batch(actions_mask, probes_n),
            attention_mask=repeat_probe_batch(attention_mask, probes_n),
            noise=noise_flat,
            time=time_flat,
        )
        flow_mse = per_probe.reshape(probes_n, bsz).mean(dim=0)

    if enable_sampled_action:
        initial_noise = noise[0]
        steps = int(sampled_action_steps)
        if steps <= 0:
            steps = int(flow.config.num_inference_steps)
        sampled = deterministic_sample_actions(
            flow,
            h_t=h_t,
            h_future=h_future,
            h_vlm=h_vlm,
            state=state,
            state_mask=state_mask,
            action_hz=action_hz,
            embodiment_id=embodiment_id,
            attention_mask=attention_mask,
            initial_noise=initial_noise,
            num_inference_steps=steps,
        )
        action_mse = masked_action_mse_per_sample(
            sampled,
            actions,
            actions_mask,
            embodiment_id,
        )
    else:
        action_mse = torch.full(
            (bsz,),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )

    return {
        "flow_mse": flow_mse.detach().cpu().float(),
        "action_mse": action_mse.detach().cpu().float(),
    }


# =============================================================================
# Probe cache
# =============================================================================

def build_probe_cache(
    base_flow,
    task_batches,
    task_ids,
    num_probes,
    seed,
    device,
    out_root,
):
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    base_flow.to(device).eval()
    print(f"\n[probes] generating {num_probes} fixed (epsilon,tau) probes per anchor")
    for task in task_ids:
        for bi, fixed_path in enumerate(task_batches[int(task)]):
            fixed = load_cache(fixed_path)
            probe_seed = int(seed) + int(task) * 100000 + int(bi) * 100
            probes = make_fixed_probes(
                base_flow,
                fixed,
                num_probes,
                probe_seed,
                device,
            )
            path = out_root / f"task_{task}" / f"batch_{bi:04d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(probes, path)
            del fixed, probes
    base_flow.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# Rows / aggregation
# =============================================================================

def fixed_meta(fixed, i):
    return {
        "trajectory_id": int(fixed["_trajectory_ids"][i]),
        "trajectory_length": int(fixed["_trajectory_lengths"][i]),
        "trajectory_step": int(fixed["_trajectory_steps"][i]),
        "phase": str(fixed["_phases"][i]),
        "phase_fraction": float(fixed["_phase_fractions"][i]),
        "instruction": str(fixed["_langs"][i]),
    }


def metric_rows_for_batch(
    *,
    suite,
    task,
    stage,
    route,
    fixed,
    metrics,
    base_metrics,
    start_ordinal,
):
    rows = []
    bsz = int(metrics["flow_mse"].shape[0])
    for i in range(bsz):
        flow_mse = float(metrics["flow_mse"][i])
        action_mse = float(metrics["action_mse"][i])
        base_flow = float(base_metrics["flow_mse"][i])
        base_action = float(base_metrics["action_mse"][i])

        rows.append(
            {
                "suite": suite,
                "task_id": int(task),
                "stage": stage,
                "route": route,
                "anchor_ordinal": int(start_ordinal + i),
                **fixed_meta(fixed, i),
                "flow_mse": flow_mse,
                "flow_rmse": math.sqrt(max(flow_mse, 0.0)),
                "flow_mse_excess_vs_base": flow_mse - base_flow,
                "action_mse": action_mse,
                "action_rmse": (
                    math.sqrt(max(action_mse, 0.0))
                    if not math.isnan(action_mse)
                    else float("nan")
                ),
                "action_mse_excess_vs_base": (
                    action_mse - base_action
                    if not math.isnan(action_mse) and not math.isnan(base_action)
                    else float("nan")
                ),
            }
        )
    return rows


def summarize_route_rows(rows):
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
        stage, route, task, traj_id = key
        out = {
            "stage": stage,
            "route": route,
            "task_id": task,
            "trajectory_id": traj_id,
            "n_anchors": len(rr),
        }
        for metric in METRICS:
            vals = [float(x[metric]) for x in rr if not math.isnan(float(x[metric]))]
            out[metric] = statistics.fmean(vals) if vals else float("nan")
        traj_rows.append(out)

    groups = {}
    for row in traj_rows:
        groups.setdefault(
            (row["stage"], row["route"], int(row["task_id"])),
            [],
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
            vals = [
                float(x[metric])
                for x in rr
                if not math.isnan(float(x[metric]))
            ]
            out[f"{metric}_mean"] = statistics.fmean(vals) if vals else float("nan")
            out[f"{metric}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))
            )
        task_rows.append(out)

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
                vals = [
                    float(x[f"{metric}_mean"])
                    for x in rr
                    if not math.isnan(float(x[f"{metric}_mean"]))
                ]
                out[f"{metric}_macro_mean"] = (
                    statistics.fmean(vals) if vals else float("nan")
                )
                out[f"{metric}_std"] = (
                    statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))
                )
            stage_rows.append(out)

    return traj_rows, task_rows, stage_rows


def build_coupling_rows(anchor_rows):
    lookup = {}
    for row in anchor_rows:
        key = (
            row["stage"],
            int(row["task_id"]),
            int(row["trajectory_id"]),
            str(row["phase"]),
            row["route"],
        )
        lookup[key] = row

    output = []
    for stage in STAGES:
        stage_keys = sorted(
            {
                (int(r["task_id"]), int(r["trajectory_id"]), str(r["phase"]))
                for r in anchor_rows
                if r["stage"] == stage
            }
        )
        for task, traj, phase in stage_keys:
            def get(route):
                return lookup[(stage, task, traj, phase, route)]

            b = get("base_ref")
            v = get("vlm_only")
            w = get("world_only")
            j = get("upstream_joint")

            flow_b = float(b["flow_mse"])
            flow_v = float(v["flow_mse"])
            flow_w = float(w["flow_mse"])
            flow_j = float(j["flow_mse"])

            action_b = float(b["action_mse"])
            action_v = float(v["action_mse"])
            action_w = float(w["action_mse"])
            action_j = float(j["action_mse"])

            output.append(
                {
                    "suite": b["suite"],
                    "stage": stage,
                    "task_id": task,
                    "trajectory_id": traj,
                    "phase": phase,
                    "phase_fraction": float(b["phase_fraction"]),
                    "vlm_excess_flow_mse": flow_v - flow_b,
                    "world_excess_flow_mse": flow_w - flow_b,
                    "upstream_joint_excess_flow_mse": flow_j - flow_b,
                    "coupling_interaction_flow_mse": (
                        flow_j - flow_v - flow_w + flow_b
                    ),
                    "vlm_excess_action_mse": (
                        action_v - action_b
                        if not math.isnan(action_b) and not math.isnan(action_v)
                        else float("nan")
                    ),
                    "world_excess_action_mse": (
                        action_w - action_b
                        if not math.isnan(action_b) and not math.isnan(action_w)
                        else float("nan")
                    ),
                    "upstream_joint_excess_action_mse": (
                        action_j - action_b
                        if not math.isnan(action_b) and not math.isnan(action_j)
                        else float("nan")
                    ),
                    "coupling_interaction_action_mse": (
                        action_j - action_v - action_w + action_b
                        if not any(math.isnan(x) for x in [action_b, action_v, action_w, action_j])
                        else float("nan")
                    ),
                }
            )
    return output


def summarize_coupling(rows):
    groups = {}
    for row in rows:
        groups.setdefault(
            (row["stage"], int(row["task_id"]), int(row["trajectory_id"])),
            [],
        ).append(row)

    traj_rows = []
    for key, rr in sorted(groups.items()):
        stage, task, traj = key
        out = {
            "stage": stage,
            "task_id": task,
            "trajectory_id": traj,
            "n_anchors": len(rr),
        }
        for metric in COUPLING_METRICS:
            vals = [float(x[metric]) for x in rr if not math.isnan(float(x[metric]))]
            out[metric] = statistics.fmean(vals) if vals else float("nan")
        traj_rows.append(out)

    groups = {}
    for row in traj_rows:
        groups.setdefault((row["stage"], int(row["task_id"])), []).append(row)

    task_rows = []
    for key, rr in sorted(groups.items()):
        stage, task = key
        out = {
            "stage": stage,
            "task_id": task,
            "n_trajectories": len(rr),
        }
        for metric in COUPLING_METRICS:
            vals = [float(x[metric]) for x in rr if not math.isnan(float(x[metric]))]
            out[f"{metric}_mean"] = statistics.fmean(vals) if vals else float("nan")
            out[f"{metric}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))
            )
        task_rows.append(out)

    stage_rows = []
    for stage in STAGES:
        rr = [x for x in task_rows if x["stage"] == stage]
        if not rr:
            continue
        out = {"stage": stage, "n_tasks": len(rr)}
        for metric in COUPLING_METRICS:
            vals = [
                float(x[f"{metric}_mean"])
                for x in rr
                if not math.isnan(float(x[f"{metric}_mean"]))
            ]
            out[f"{metric}_macro_mean"] = (
                statistics.fmean(vals) if vals else float("nan")
            )
            out[f"{metric}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))
            )
        stage_rows.append(out)

    return traj_rows, task_rows, stage_rows


def write_matrix(path, task_rows, task_ids, *, route, metric):
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


def write_coupling_matrix(path, task_rows, task_ids, *, metric):
    lookup = {
        (x["stage"], int(x["task_id"])): float(x[f"{metric}_mean"])
        for x in task_rows
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
    lookup = {(x["stage"], x["route"]): x for x in stage_rows}
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


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    sem.seed_all(args.seed)
    args.trajectory_fractions = sem.validate_fractions(args.trajectory_fractions)

    if args.num_flow_probes <= 0:
        raise ValueError("--num-flow-probes must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

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

    print("=" * 96)
    print("LaWAM Action Generation / Conditioning-Coupling Forgetting")
    print("Fixed-flow-probe + 6-route intervention")
    print("=" * 96)
    print("suite                 :", args.suite)
    print("Base tasks            :", args.task_ids)
    print("fractions             :", args.trajectory_fractions)
    print("split                 :", args.split)
    print("max trajectories/task :", args.max_trajectories_per_task)
    print("flow probes / anchor  :", args.num_flow_probes)
    print("sampled action aux    :", args.enable_sampled_action)
    for stage in STAGES:
        print(f"{stage:4s}: {chain[stage]}")
    print("=" * 96)

    run_meta = {
        "suite": args.suite,
        "task_ids": [int(x) for x in args.task_ids],
        "split": args.split,
        "trajectory_fractions": [float(x) for x in args.trajectory_fractions],
        "num_flow_probes": int(args.num_flow_probes),
        "enable_sampled_action": bool(args.enable_sampled_action),
        "sampled_action_steps": int(args.sampled_action_steps),
        "checkpoints": {k: str(v) for k, v in chain.items()},
        "routes": {
            "base_ref": "Flow_Base(H_VLM_Base, future_Base)",
            "vlm_only": "Flow_Base(H_VLM_CLk, future_Base)",
            "world_only": "Flow_Base(H_VLM_Base, future_CLk)",
            "upstream_joint": "Flow_Base(H_VLM_CLk, future_CLk)",
            "flow_only": "Flow_CLk(H_VLM_Base, future_Base)",
            "full": "Flow_CLk(H_VLM_CLk, future_CLk)",
        },
        "primary_metric": (
            "Masked MSE between predicted and target flow velocity at fixed "
            "(epsilon,tau), averaged over fixed probes."
        ),
        "coupling_interaction": (
            "E(VLM_CL,Future_CL)-E(VLM_CL,Future_B)-E(VLM_B,Future_CL)+E(Base) "
            "with Base Flow fixed."
        ),
        "aggregation": (
            "probe mean per anchor -> 25/50/75 mean within trajectory -> "
            "equal trajectory mean within task -> equal task macro mean."
        ),
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Sanity checks: Flow and LaWM decoder are both trainable in the LIBERO CL chain.
    write_module_change_check(
        chain,
        FLOW_PREFIX,
        args.output_dir / "flow_change_check.csv",
        "flow",
    )
    write_module_change_check(
        chain,
        DECODER_PREFIX,
        args.output_dir / "decoder_change_check.csv",
        "decoder",
    )

    # Fixed trajectory-balanced action inputs.
    collator = sem.build_eval_collator(cfg)
    input_root = args.output_dir / "fixed_inputs"
    task_batches = {}
    coverage = []

    for task in args.task_ids:
        paths, meta = materialize_action_task(
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

    # Permanent Base Flow; CL Flow will be loaded into backend.flow stage by stage.
    base_flow = copy.deepcopy(backend.flow).cpu().eval()
    for p in base_flow.parameters():
        p.requires_grad_(False)

    # Base conditions: actual Base H_VLM and actual Base predicted future.
    base_condition_root = args.output_dir / "base_conditions"
    build_base_condition_cache(
        backend,
        task_batches,
        args.task_ids,
        device,
        base_condition_root,
    )

    # Exact fixed noise/time probes shared by ALL stages and routes.
    probe_root = args.output_dir / "fixed_flow_probes"
    build_probe_cache(
        base_flow,
        task_batches,
        args.task_ids,
        args.num_flow_probes,
        args.seed + 700000,
        device,
        probe_root,
    )

    # Base metrics.
    anchor_rows = []
    base_metrics_cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    print("\n[Base] evaluating fixed-flow reference")
    base_flow.to(device).eval()
    for task in args.task_ids:
        ordinal = 0
        for bi, fixed_path in enumerate(task_batches[int(task)]):
            fixed = load_cache(fixed_path)
            base_cond = load_cache(
                base_condition_root / f"task_{task}" / f"batch_{bi:04d}.pt"
            )
            probes = load_cache(
                probe_root / f"task_{task}" / f"batch_{bi:04d}.pt"
            )
            metrics = evaluate_route(
                base_flow,
                fixed=fixed,
                h_t=base_cond["h_t"],
                h_vlm=base_cond["h_vlm"],
                h_future=base_cond["h_future"],
                probes=probes,
                device=device,
                enable_sampled_action=args.enable_sampled_action,
                sampled_action_steps=args.sampled_action_steps,
            )
            base_metrics_cache[(int(task), int(bi))] = {
                k: v.clone()
                for k, v in metrics.items()
            }

            # At Base, all six intervention routes collapse to the same computation.
            for route in ROUTES:
                anchor_rows += metric_rows_for_batch(
                    suite=args.suite,
                    task=int(task),
                    stage="Base",
                    route=route,
                    fixed=fixed,
                    metrics=metrics,
                    base_metrics=metrics,
                    start_ordinal=ordinal,
                )
            ordinal += int(metrics["flow_mse"].shape[0])
            del fixed, base_cond, probes, metrics

        print(f"[Base] task={task}: {ordinal} anchors")

    base_flow.to("cpu")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # CL stages.
    stage_condition_parent = args.output_dir / "_stage_conditions"

    for stage in STAGES[1:]:
        print(f"\n[{stage}] building CL H_VLM + predicted-future conditions")

        # Load stage VLM/QFormer/decoder/flow while modules are on CPU.
        lat.load_stage_vlm_qformer(model, chain[stage])
        load_stage_decoder(model, chain[stage])
        load_stage_flow(model, chain[stage])

        stage_root = stage_condition_parent / stage
        build_stage_condition_cache(
            model,
            chain[stage],
            task_batches,
            args.task_ids,
            base_condition_root,
            device,
            stage_root,
        )

        # -------------------------------------------------------------
        # Base Flow routes: direct VLM, world, and joint upstream effects.
        # -------------------------------------------------------------
        print(f"[{stage}] evaluating Base-Flow upstream interventions")
        base_flow.to(device).eval()

        stage_route_metrics: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}

        for task in args.task_ids:
            ordinal = 0
            for bi, fixed_path in enumerate(task_batches[int(task)]):
                fixed = load_cache(fixed_path)
                base_cond = load_cache(
                    base_condition_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )
                cl_cond = load_cache(
                    stage_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )
                probes = load_cache(
                    probe_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )
                base_metrics = base_metrics_cache[(int(task), int(bi))]

                route_conditions = {
                    "vlm_only": (
                        cl_cond["h_vlm"],
                        base_cond["h_future"],
                    ),
                    "world_only": (
                        base_cond["h_vlm"],
                        cl_cond["h_future"],
                    ),
                    "upstream_joint": (
                        cl_cond["h_vlm"],
                        cl_cond["h_future"],
                    ),
                }

                # base_ref is a cached invariant route.
                anchor_rows += metric_rows_for_batch(
                    suite=args.suite,
                    task=int(task),
                    stage=stage,
                    route="base_ref",
                    fixed=fixed,
                    metrics=base_metrics,
                    base_metrics=base_metrics,
                    start_ordinal=ordinal,
                )

                for route, (h_vlm, h_future) in route_conditions.items():
                    metrics = evaluate_route(
                        base_flow,
                        fixed=fixed,
                        h_t=base_cond["h_t"],
                        h_vlm=h_vlm,
                        h_future=h_future,
                        probes=probes,
                        device=device,
                        enable_sampled_action=args.enable_sampled_action,
                        sampled_action_steps=args.sampled_action_steps,
                    )
                    stage_route_metrics[(int(task), int(bi), route)] = {
                        k: v.clone()
                        for k, v in metrics.items()
                    }
                    anchor_rows += metric_rows_for_batch(
                        suite=args.suite,
                        task=int(task),
                        stage=stage,
                        route=route,
                        fixed=fixed,
                        metrics=metrics,
                        base_metrics=base_metrics,
                        start_ordinal=ordinal,
                    )
                    del metrics

                ordinal += int(base_metrics["flow_mse"].shape[0])
                del fixed, base_cond, cl_cond, probes

            print(f"[{stage}] Base-Flow task={task}: {ordinal} anchors")

        base_flow.to("cpu")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # CL Flow routes: Flow-only and full.
        # -------------------------------------------------------------
        print(f"[{stage}] evaluating CL-Flow interventions")
        backend.flow.to(device).eval()

        for task in args.task_ids:
            ordinal = 0
            for bi, fixed_path in enumerate(task_batches[int(task)]):
                fixed = load_cache(fixed_path)
                base_cond = load_cache(
                    base_condition_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )
                cl_cond = load_cache(
                    stage_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )
                probes = load_cache(
                    probe_root / f"task_{task}" / f"batch_{bi:04d}.pt"
                )
                base_metrics = base_metrics_cache[(int(task), int(bi))]

                for route, (h_vlm, h_future) in {
                    "flow_only": (
                        base_cond["h_vlm"],
                        base_cond["h_future"],
                    ),
                    "full": (
                        cl_cond["h_vlm"],
                        cl_cond["h_future"],
                    ),
                }.items():
                    metrics = evaluate_route(
                        backend.flow,
                        fixed=fixed,
                        h_t=base_cond["h_t"],
                        h_vlm=h_vlm,
                        h_future=h_future,
                        probes=probes,
                        device=device,
                        enable_sampled_action=args.enable_sampled_action,
                        sampled_action_steps=args.sampled_action_steps,
                    )
                    anchor_rows += metric_rows_for_batch(
                        suite=args.suite,
                        task=int(task),
                        stage=stage,
                        route=route,
                        fixed=fixed,
                        metrics=metrics,
                        base_metrics=base_metrics,
                        start_ordinal=ordinal,
                    )
                    del metrics

                ordinal += int(base_metrics["flow_mse"].shape[0])
                del fixed, base_cond, cl_cond, probes

            print(f"[{stage}] CL-Flow task={task}: {ordinal} anchors")

        backend.flow.to("cpu")
        del stage_route_metrics
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if not args.keep_stage_condition_cache:
            shutil.rmtree(stage_root, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Save route-level results.
    # -------------------------------------------------------------------------
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
        args.output_dir / "action_anchor_metrics.csv",
        anchor_rows,
        anchor_fields,
    )

    traj_rows, task_rows, stage_rows = summarize_route_rows(anchor_rows)

    sem.write_csv(
        args.output_dir / "action_trajectory_summary.csv",
        traj_rows,
        ["stage", "route", "task_id", "trajectory_id", "n_anchors", *METRICS],
    )

    task_fields = ["stage", "route", "task_id", "n_trajectories"]
    for metric in METRICS:
        task_fields += [f"{metric}_mean", f"{metric}_std"]
    sem.write_csv(
        args.output_dir / "action_task_summary.csv",
        task_rows,
        task_fields,
    )

    stage_fields = ["stage", "route", "n_tasks"]
    for metric in METRICS:
        stage_fields += [f"{metric}_macro_mean", f"{metric}_std"]
    sem.write_csv(
        args.output_dir / "action_stage_summary.csv",
        stage_rows,
        stage_fields,
    )

    write_intervention_summary(
        args.output_dir / "action_intervention_stage_summary.csv",
        stage_rows,
    )

    # -------------------------------------------------------------------------
    # Explicit VLM-world coupling interaction under Base Flow.
    # -------------------------------------------------------------------------
    coupling_anchor = build_coupling_rows(anchor_rows)
    coupling_traj, coupling_task, coupling_stage = summarize_coupling(coupling_anchor)

    sem.write_csv(
        args.output_dir / "action_coupling_anchor_summary.csv",
        coupling_anchor,
        [
            "suite",
            "stage",
            "task_id",
            "trajectory_id",
            "phase",
            "phase_fraction",
            *COUPLING_METRICS,
        ],
    )

    coupling_task_fields = ["stage", "task_id", "n_trajectories"]
    for metric in COUPLING_METRICS:
        coupling_task_fields += [f"{metric}_mean", f"{metric}_std"]
    sem.write_csv(
        args.output_dir / "action_coupling_task_summary.csv",
        coupling_task,
        coupling_task_fields,
    )

    coupling_stage_fields = ["stage", "n_tasks"]
    for metric in COUPLING_METRICS:
        coupling_stage_fields += [f"{metric}_macro_mean", f"{metric}_std"]
    sem.write_csv(
        args.output_dir / "action_coupling_stage_summary.csv",
        coupling_stage,
        coupling_stage_fields,
    )

    # Core flow-MSE matrices.
    core_flow_routes = [
        "full",
        "flow_only",
        "upstream_joint",
        "vlm_only",
        "world_only",
    ]
    for route in core_flow_routes:
        write_matrix(
            args.output_dir / f"matrix_{route}_flow_mse.csv",
            task_rows,
            args.task_ids,
            route=route,
            metric="flow_mse",
        )
        write_matrix(
            args.output_dir / f"matrix_{route}_flow_mse_excess_vs_base.csv",
            task_rows,
            args.task_ids,
            route=route,
            metric="flow_mse_excess_vs_base",
        )

    write_coupling_matrix(
        args.output_dir / "matrix_coupling_interaction_flow_mse.csv",
        coupling_task,
        args.task_ids,
        metric="coupling_interaction_flow_mse",
    )

    if args.enable_sampled_action:
        for route in core_flow_routes:
            write_matrix(
                args.output_dir / f"matrix_{route}_action_mse.csv",
                task_rows,
                args.task_ids,
                route=route,
                metric="action_mse",
            )
            write_matrix(
                args.output_dir / f"matrix_{route}_action_mse_excess_vs_base.csv",
                task_rows,
                args.task_ids,
                route=route,
                metric="action_mse_excess_vs_base",
            )
        write_coupling_matrix(
            args.output_dir / "matrix_coupling_interaction_action_mse.csv",
            coupling_task,
            args.task_ids,
            metric="coupling_interaction_action_mse",
        )

    print("\n" + "=" * 96)
    print("Action Generation / Coupling Forgetting analysis complete")
    print("=" * 96)
    print("flow check   :", args.output_dir / "flow_change_check.csv")
    print("decoder check:", args.output_dir / "decoder_change_check.csv")
    print("intervention :", args.output_dir / "action_intervention_stage_summary.csv")
    print("coupling     :", args.output_dir / "action_coupling_stage_summary.csv")
    print("core matrices:")
    for route in core_flow_routes:
        print(" ", args.output_dir / f"matrix_{route}_flow_mse.csv")
    print(" ", args.output_dir / "matrix_coupling_interaction_flow_mse.csv")
    if args.enable_sampled_action:
        print("sampled-action auxiliary matrices were also generated.")
    print("=" * 96)


if __name__ == "__main__":
    main()