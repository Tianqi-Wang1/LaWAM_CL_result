#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from starVLA.dataloader.latent_world_train_collator import LatentWorldTrainCollator
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.processor_utils import build_latent_world_processor_spec
from starVLA.model.framework.lawam_framework import LaWAMFramework


STAGES = ["Base", "CL1", "CL2", "CL3", "CL4"]
RUN_PATTERNS = {
    "Base": "*+base_t0_5_10k_4gpu_bs32_ga2",
    "CL1": "*+cl1_t6_2k_4gpu_bs32_ga2",
    "CL2": "*+cl2_t7_2k_4gpu_bs32_ga2",
    "CL3": "*+cl3_t8_2k_4gpu_bs32_ga2",
    "CL4": "*+cl4_t9_2k_4gpu_bs32_ga2",
}

VLM_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "act_placeholder_mask",
    "flow_placeholder_mask",
)

METRICS = (
    "context_rel_l2",
    "context_cosine",
    "conditioning_rel_l2",
    "conditioning_cosine",
)


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Trajectory-balanced LaWAM Semantic / Conditioning Forgetting analysis. "
            "For every Base-task demonstration trajectory, sample fixed progress anchors "
            "(default 25/50/75%) and compare Base vs CL1-CL4 h_VLM features."
        )
    )
    p.add_argument("--suite", required=True, choices=["libero_goal", "libero_object"])
    p.add_argument(
        "--config-yaml",
        type=Path,
        default=Path("starVLA/config/training/train_libero.yaml"),
        help="Canonical full LaWAM LIBERO config.",
    )
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument(
        "--trajectory-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="Trajectory progress fractions used as fixed anchors.",
    )
    p.add_argument(
        "--max-trajectories-per-task",
        type=int,
        default=0,
        help="0 means use every filtered demonstration trajectory. Positive values are for smoke/debug only.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Formal protocol should use all.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--reuse-input-cache", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_fractions(fractions: Sequence[float]) -> list[float]:
    vals = [float(x) for x in fractions]
    if not vals:
        raise ValueError("--trajectory-fractions cannot be empty")
    for x in vals:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"trajectory fraction must be in [0,1], got {x}")
    if len(set(vals)) != len(vals):
        raise ValueError(f"trajectory fractions must be unique, got {vals}")
    return vals


def phase_label(fraction: float) -> str:
    return f"p{int(round(float(fraction) * 100)):02d}"


def resolve_chain(run_root: Path) -> dict[str, Path]:
    chain: dict[str, Path] = {}
    for stage in STAGES:
        runs = sorted(x for x in run_root.glob(RUN_PATTERNS[stage]) if x.is_dir())
        if not runs:
            raise FileNotFoundError(
                f"No {stage} run under {run_root}: {RUN_PATTERNS[stage]}"
            )
        run = runs[-1]
        ckpt = run / "final_model" / "pytorch_model.pt"
        for path in [ckpt, run / "config.yaml", run / "dataset_statistics.json"]:
            if not path.is_file():
                raise FileNotFoundError(path)
        chain[stage] = ckpt
    return chain


def run_dir(ckpt: Path) -> Path:
    return ckpt.parents[1]


def load_base_cfg_stats(base_ckpt: Path, canonical_config_yaml: Path):
    """Merge canonical config with the saved Base run config.

    LaWAM checkpoint saving can leave an access-tracked run config that omits
    dataset-only keys.  The canonical config restores the complete dataset schema,
    while the Base run config still wins for run-specific values that are present.
    """
    rd = run_dir(base_ckpt)
    canonical_config_yaml = canonical_config_yaml.expanduser().resolve()
    if not canonical_config_yaml.is_file():
        raise FileNotFoundError(f"Canonical config not found: {canonical_config_yaml}")

    canonical_cfg = OmegaConf.load(canonical_config_yaml)
    run_cfg = OmegaConf.load(rd / "config.yaml")
    cfg = OmegaConf.merge(canonical_cfg, run_cfg)

    required = ("data_root_dir", "data_mix", "image_resolution", "num_frames", "sec_chunk")
    missing = [k for k in required if cfg.datasets.vla_data.get(k, None) is None]
    if missing:
        raise RuntimeError(
            "Analysis dataset config remains incomplete after canonical + run merge. "
            f"Missing={missing}; canonical={canonical_config_yaml}; run={rd / 'config.yaml'}"
        )

    with (rd / "dataset_statistics.json").open("r", encoding="utf-8") as f:
        stats = json.load(f)

    print("[config] canonical :", canonical_config_yaml)
    print("[config] Base run  :", rd / "config.yaml")
    print("[config] data_root :", cfg.datasets.vla_data.data_root_dir)
    print("[config] data_mix  :", cfg.datasets.vla_data.data_mix)
    return cfg, stats


def build_eval_collator(cfg) -> LatentWorldTrainCollator:
    policy_cfg = LatentWorldPolicyConfigBuilder(cfg).build()
    spec = build_latent_world_processor_spec(
        policy_cfg=policy_cfg,
        vlm_model_id=str(cfg.framework.qwenvl.base_vlm),
    )
    dc = cfg.datasets.vla_data
    return LatentWorldTrainCollator(
        policy_cfg=policy_cfg,
        processor_spec=spec,
        act_queries=int(policy_cfg.num_action_queries),
        flow_queries=int(policy_cfg.flow_action_num_queries),
        enable_primary_video_aug=bool(dc.get("enable_primary_video_aug", False)),
        enable_primary_random_resized_crop=bool(
            dc.get("enable_primary_random_resized_crop", False)
        ),
        cot_prompt_before_wrist=dc.get("CoT_prompt_before_wrist", None),
        cot_prompt_after_wrist=dc.get("CoT_prompt_after_wrist", None),
    ).eval()


def task_data_cfg(cfg, suite: str, task_id: int):
    dc = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    dc.cl_suite = suite
    dc.cl_task_ids = [int(task_id)]
    dc.use_task_filtered_statistics = False
    for key in ("data_root_dir", "data_mix"):
        if dc.get(key, None) is None:
            raise RuntimeError(f"task_data_cfg missing required key {key!r}")
    return dc


def worker_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def choose_trajectory_indices(total: int, max_trajectories: int) -> list[int]:
    """Use every trajectory formally; choose evenly spread trajectories for smoke."""
    if total <= 0:
        return []
    if max_trajectories <= 0 or max_trajectories >= total:
        return list(range(total))
    raw = np.linspace(0, total - 1, num=max_trajectories)
    chosen: list[int] = []
    for x in raw:
        idx = int(round(float(x)))
        if idx not in chosen:
            chosen.append(idx)
    # Rounding can only reduce count in pathological tiny cases. Fill deterministically.
    if len(chosen) < max_trajectories:
        for idx in range(total):
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == max_trajectories:
                break
    return sorted(chosen)


def build_anchor_specs(single_dataset, fractions: Sequence[float], max_trajectories: int):
    traj_ids = np.asarray(single_dataset.trajectory_ids, dtype=np.int64)
    traj_lengths = np.asarray(single_dataset.trajectory_lengths, dtype=np.int64)
    if len(traj_ids) != len(traj_lengths):
        raise RuntimeError("trajectory_ids / trajectory_lengths size mismatch")

    selected_traj_indices = choose_trajectory_indices(
        len(traj_ids), int(max_trajectories)
    )

    specs: list[dict[str, Any]] = []
    warnings: list[str] = []
    for traj_local_idx in selected_traj_indices:
        trajectory_id = int(traj_ids[traj_local_idx])
        length = int(traj_lengths[traj_local_idx])
        if length <= 0:
            warnings.append(f"trajectory {trajectory_id} has non-positive length={length}; skipped")
            continue

        steps = [int(round((length - 1) * float(frac))) for frac in fractions]
        if len(set(steps)) != len(steps):
            warnings.append(
                f"trajectory {trajectory_id} length={length} maps multiple phases to the same step: {steps}"
            )

        for fraction, step in zip(fractions, steps):
            specs.append(
                {
                    "trajectory_local_index": int(traj_local_idx),
                    "trajectory_id": trajectory_id,
                    "trajectory_length": length,
                    "trajectory_step": int(step),
                    "phase_fraction": float(fraction),
                    "phase": phase_label(float(fraction)),
                }
            )

    return specs, selected_traj_indices, warnings


class TrajectoryAnchorDataset(Dataset):
    """Deterministically materialize exact (trajectory_id, step) samples.

    get_vla_dataset() returns a LeRobotMixtureDataset.  After LIBERO task filtering
    our analysis expects exactly one underlying LeRobotSingleDataset.  We reuse the
    mixture's own transforms and output builder, but bypass its random step sampler
    so the requested 25/50/75% anchors are exact and trajectory-balanced.
    """

    def __init__(self, mixture_dataset, specs: Sequence[dict[str, Any]]):
        self.mixture = mixture_dataset
        datasets = list(getattr(mixture_dataset, "datasets", []))
        if len(datasets) != 1:
            raise RuntimeError(
                "Trajectory-balanced LIBERO analysis expects exactly one underlying "
                f"dataset after task filtering, got {len(datasets)}."
            )
        self.single = datasets[0]
        self.specs = list(specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int):
        spec = self.specs[int(index)]

        # Preserve the mixture's view-selection semantics. For LIBERO this normally
        # resolves to the configured primary/wrist views; if random view selection is
        # enabled, it is still deterministic for this fixed anchor index.
        selected_video_keys = self.mixture._select_video_keys_for_sample(
            self.single, int(index)
        )
        if not selected_video_keys:
            raise ValueError("No video keys resolved for trajectory anchor")

        raw_data = self.single.get_step_data(
            int(spec["trajectory_id"]),
            int(spec["trajectory_step"]),
            modality_keys_override={"video": selected_video_keys},
        )
        transforms = self.mixture._get_transforms_for_selected_video_keys(
            self.single, selected_video_keys
        )
        data = transforms(raw_data)
        sample = self.mixture._build_output_sample(
            self.single, data, selected_video_keys
        )
        return spec, sample


class AnchorCollator:
    def __init__(self, collator: LatentWorldTrainCollator):
        self.collator = collator

    def __call__(self, items):
        specs, samples = zip(*items)
        batch = self.collator(samples)
        batch["_trajectory_ids"] = [int(x["trajectory_id"]) for x in specs]
        batch["_trajectory_lengths"] = [int(x["trajectory_length"]) for x in specs]
        batch["_trajectory_steps"] = [int(x["trajectory_step"]) for x in specs]
        batch["_phase_fractions"] = [float(x["phase_fraction"]) for x in specs]
        batch["_phases"] = [str(x["phase"]) for x in specs]
        batch["_langs"] = [str(x["lang"]) for x in samples]
        return batch


def slim_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in VLM_KEYS:
        value = batch[key]
        out[key] = value.detach().cpu().contiguous() if torch.is_tensor(value) else value
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


def write_anchor_manifest(path: Path, suite: str, task_id: int, specs: Sequence[dict[str, Any]]):
    rows = []
    for ordinal, spec in enumerate(specs):
        rows.append(
            {
                "suite": suite,
                "task_id": int(task_id),
                "anchor_ordinal": int(ordinal),
                "trajectory_local_index": int(spec["trajectory_local_index"]),
                "trajectory_id": int(spec["trajectory_id"]),
                "trajectory_length": int(spec["trajectory_length"]),
                "trajectory_step": int(spec["trajectory_step"]),
                "phase": str(spec["phase"]),
                "phase_fraction": float(spec["phase_fraction"]),
            }
        )
    write_csv(
        path,
        rows,
        [
            "suite",
            "task_id",
            "anchor_ordinal",
            "trajectory_local_index",
            "trajectory_id",
            "trajectory_length",
            "trajectory_step",
            "phase",
            "phase_fraction",
        ],
    )


def cache_compatible(
    meta: dict[str, Any],
    *,
    suite: str,
    task_id: int,
    split: str,
    fractions: Sequence[float],
    max_trajectories: int,
    batch_size: int,
) -> bool:
    cached_fractions = [float(x) for x in meta.get("trajectory_fractions", [])]
    return (
        meta.get("suite") == suite
        and int(meta.get("task_id", -1)) == int(task_id)
        and meta.get("split") == split
        and cached_fractions == [float(x) for x in fractions]
        and int(meta.get("max_trajectories_per_task", -999)) == int(max_trajectories)
        and int(meta.get("batch_size", -1)) == int(batch_size)
        and int(meta.get("num_batches", 0)) > 0
    )


def materialize_task_inputs(
    *,
    cfg,
    stats,
    collator,
    suite: str,
    task_id: int,
    split: str,
    fractions: Sequence[float],
    max_trajectories: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    cache_root: Path,
    reuse: bool,
):
    td = cache_root / f"task_{task_id}"
    meta_path = td / "meta.json"

    if reuse and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        paths = sorted(td.glob("batch_*.pt"))
        if cache_compatible(
            meta,
            suite=suite,
            task_id=task_id,
            split=split,
            fractions=fractions,
            max_trajectories=max_trajectories,
            batch_size=batch_size,
        ) and len(paths) == int(meta.get("num_batches", 0)):
            print(
                f"[cache] task={task_id}: reuse trajectories={meta['num_trajectories']} "
                f"anchors={meta['num_samples']}"
            )
            return paths, meta

    if td.exists():
        shutil.rmtree(td)
    td.mkdir(parents=True, exist_ok=True)

    mixture = get_vla_dataset(
        data_cfg=task_data_cfg(cfg, suite, task_id),
        mode=split,
        balance_dataset_weights=False,
        seed=seed,
        framework_name=str(cfg.framework.name),
        dataset_statistics_override=stats,
    )

    # mode=all gives full trajectory coverage.  Upstream treats train/all transforms
    # as training transforms, so force deterministic eval transforms for this probe.
    for single_dataset in getattr(mixture, "datasets", []):
        transforms = getattr(single_dataset, "transforms", None)
        if transforms is not None and hasattr(transforms, "eval"):
            transforms.eval()

    datasets = list(getattr(mixture, "datasets", []))
    if len(datasets) != 1:
        raise RuntimeError(
            f"Expected one filtered LIBERO dataset, got {len(datasets)} for {suite} task={task_id}"
        )
    single = datasets[0]

    specs, selected_traj_indices, warnings = build_anchor_specs(
        single,
        fractions=fractions,
        max_trajectories=max_trajectories,
    )
    if not specs:
        raise RuntimeError(f"No trajectory anchors for {suite} task={task_id}")

    for warning in warnings:
        print("[WARN]", warning)

    anchor_dataset = TrajectoryAnchorDataset(mixture, specs)
    generator = torch.Generator().manual_seed(seed + task_id)
    kwargs: dict[str, Any] = {
        "dataset": anchor_dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "collate_fn": AnchorCollator(collator),
        "num_workers": int(num_workers),
        "pin_memory": False,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init
        kwargs["persistent_workers"] = False
    loader = DataLoader(**kwargs)

    paths: list[Path] = []
    count = 0
    for batch_idx, batch in enumerate(loader):
        path = td / f"batch_{batch_idx:04d}.pt"
        fixed = slim_batch(batch)
        torch.save(fixed, path)
        paths.append(path)
        count += int(fixed["input_ids"].shape[0])

    num_traj = len(selected_traj_indices)
    expected = num_traj * len(fractions)
    if count != expected:
        raise RuntimeError(
            f"Anchor count mismatch for task={task_id}: wrote={count}, expected={expected}"
        )

    meta = {
        "suite": suite,
        "task_id": int(task_id),
        "split": split,
        "trajectory_fractions": [float(x) for x in fractions],
        "phase_labels": [phase_label(x) for x in fractions],
        "max_trajectories_per_task": int(max_trajectories),
        "available_trajectories": int(len(single.trajectory_ids)),
        "available_steps": int(len(single)),
        "selected_trajectory_local_indices": [int(x) for x in selected_traj_indices],
        "selected_trajectory_ids": sorted({int(x["trajectory_id"]) for x in specs}),
        "num_trajectories": int(num_traj),
        "num_samples": int(count),
        "batch_size": int(batch_size),
        "num_batches": int(len(paths)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_anchor_manifest(td / "anchor_manifest.csv", suite, task_id, specs)

    print(
        f"[anchors] suite={suite} task={task_id} split={split} "
        f"trajectories={num_traj}/{len(single.trajectory_ids)} "
        f"fractions={[phase_label(x) for x in fractions]} anchors={count}"
    )
    print(f"[cache] task={task_id}: wrote {count} fixed VLM inputs")
    return paths, meta


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(state):
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_vlm_queries(model, ckpt: Path) -> None:
    """Swap only parameters that can affect h_VLM: VLM + learned queries."""
    state = normalize_state_dict(torch_load(ckpt))
    backend = model.policy_backend
    prefix = "policy_backend.vlm."
    vlm_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not vlm_state:
        raise RuntimeError(f"No VLM keys in {ckpt}")

    result = backend.vlm.load_state_dict(vlm_state, strict=False)
    missing = [k for k in result.missing_keys if "lm_head" not in k.split(".")]
    unexpected = [k for k in result.unexpected_keys if "lm_head" not in k.split(".")]
    if missing or unexpected:
        raise RuntimeError(
            f"VLM mismatch for {ckpt}\nmissing={missing[:20]}\nunexpected={unexpected[:20]}"
        )

    for key in ("policy_backend.act_query", "policy_backend.flow_action_query"):
        if key not in state:
            raise RuntimeError(f"Missing {key} in {ckpt}")

    with torch.no_grad():
        backend.act_query.copy_(state["policy_backend.act_query"].to(backend.act_query))
        backend.flow_action_query.copy_(
            state["policy_backend.flow_action_query"].to(backend.flow_action_query)
        )

    del vlm_state, state
    gc.collect()


def to_device_batch(fixed: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in VLM_KEYS:
        value = fixed[key]
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def vlm_hidden_only(backend, fixed: dict[str, Any], device: torch.device):
    """Mirror LaWAM's VLM stage and stop at last_hidden_state."""
    batch = to_device_batch(fixed, device)
    dtype = backend.model_cfg.vlm_dtype
    act_q, flow_q = backend._prepare_queries(device=device, vlm_stage_dtype=dtype)
    autocast_ctx = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )

    with torch.inference_mode(), autocast_ctx:
        embed = backend.vlm.get_input_embeddings()
        if embed is None:
            raise RuntimeError("VLM input embedding layer is unavailable")
        inputs_embeds = embed(batch["input_ids"])
        backend._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=batch["act_placeholder_mask"],
            queries=act_q,
            num_queries=int(backend.num_action_queries),
            name="act_query",
        )
        backend._inject_queries(
            inputs_embeds=inputs_embeds,
            placeholder_mask=batch["flow_placeholder_mask"],
            queries=flow_q,
            num_queries=int(flow_q.shape[0]),
            name="flow_query",
        )
        out = backend.vlm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
        )
        hidden = out.last_hidden_state

    return hidden.detach(), batch["attention_mask"], batch["act_placeholder_mask"]


def select_features(hidden, attention_mask, act_mask, num_q: int):
    bsz, seq_len, hidden_dim = hidden.shape
    act_mask = act_mask.bool()
    valid = attention_mask.bool()
    counts = act_mask.sum(1)
    if not torch.all(counts == int(num_q)):
        raise RuntimeError(
            f"act query count mismatch: {counts.detach().cpu().tolist()} expected={num_q}"
        )

    conditioning = hidden[act_mask].view(bsz, num_q, hidden_dim)

    positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(bsz, -1)
    sentinel = torch.full_like(positions, seq_len)
    first_act = torch.where(act_mask, positions, sentinel).min(1).values
    context_mask = valid & (positions < first_act[:, None])
    max_len = int(first_act.max().item())
    context = hidden[:, :max_len]
    context_mask = context_mask[:, :max_len]

    if not torch.all(context_mask.any(dim=1)):
        raise RuntimeError("At least one sample has empty pre-act context")
    return context, context_mask, conditioning


def rel_l2(cur, base, eps: float = 1e-8) -> float:
    cur = cur.float()
    base = base.float()
    numerator = torch.linalg.vector_norm(cur - base)
    denominator = torch.linalg.vector_norm(base).clamp_min(eps)
    return float((numerator / denominator).item())


def token_cos(cur, base) -> float:
    return float(
        F.cosine_similarity(cur.float(), base.float(), dim=-1, eps=1e-8)
        .mean()
        .item()
    )


def save_base_feature(path: Path, context, mask, cond, fixed) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "context": context.detach().cpu().to(torch.bfloat16),
            "context_mask": mask.detach().cpu().bool(),
            "conditioning": cond.detach().cpu().to(torch.bfloat16),
            "trajectory_ids": list(fixed["_trajectory_ids"]),
            "trajectory_lengths": list(fixed["_trajectory_lengths"]),
            "trajectory_steps": list(fixed["_trajectory_steps"]),
            "phase_fractions": list(fixed["_phase_fractions"]),
            "phases": list(fixed["_phases"]),
            "langs": list(fixed["_langs"]),
        },
        path,
    )


def row_metadata(base: dict[str, Any], i: int) -> dict[str, Any]:
    return {
        "trajectory_id": int(base["trajectory_ids"][i]),
        "trajectory_length": int(base["trajectory_lengths"][i]),
        "trajectory_step": int(base["trajectory_steps"][i]),
        "phase": str(base["phases"][i]),
        "phase_fraction": float(base["phase_fractions"][i]),
        "instruction": str(base["langs"][i]),
    }


def compare_batch(suite, task, stage, base, context, mask, cond, start):
    context = context.detach().cpu().to(torch.bfloat16)
    mask = mask.detach().cpu().bool()
    cond = cond.detach().cpu().to(torch.bfloat16)

    if context.shape != base["context"].shape:
        raise RuntimeError(
            f"context shape changed in {stage}, task={task}: "
            f"base={tuple(base['context'].shape)} current={tuple(context.shape)}"
        )
    if cond.shape != base["conditioning"].shape:
        raise RuntimeError(
            f"conditioning shape changed in {stage}, task={task}: "
            f"base={tuple(base['conditioning'].shape)} current={tuple(cond.shape)}"
        )
    if not torch.equal(mask, base["context_mask"].bool()):
        raise RuntimeError(f"context mask changed in {stage}, task={task}")

    rows = []
    for i in range(context.shape[0]):
        valid = mask[i]
        cur_context = context[i][valid]
        base_context = base["context"][i][valid]
        cur_cond = cond[i]
        base_cond = base["conditioning"][i]
        meta = row_metadata(base, i)
        rows.append(
            {
                "suite": suite,
                "task_id": int(task),
                "stage": stage,
                "anchor_ordinal": int(start + i),
                **meta,
                "context_tokens": int(valid.sum().item()),
                "conditioning_tokens": int(cur_cond.shape[0]),
                "context_rel_l2": rel_l2(cur_context, base_context),
                "context_cosine": token_cos(cur_context, base_context),
                "conditioning_rel_l2": rel_l2(cur_cond, base_cond),
                "conditioning_cosine": token_cos(cur_cond, base_cond),
            }
        )
    return rows


def identity_rows(suite, task, base, start):
    rows = []
    mask = base["context_mask"].bool()
    for i in range(mask.shape[0]):
        meta = row_metadata(base, i)
        rows.append(
            {
                "suite": suite,
                "task_id": int(task),
                "stage": "Base",
                "anchor_ordinal": int(start + i),
                **meta,
                "context_tokens": int(mask[i].sum().item()),
                "conditioning_tokens": int(base["conditioning"][i].shape[0]),
                "context_rel_l2": 0.0,
                "context_cosine": 1.0,
                "conditioning_rel_l2": 0.0,
                "conditioning_cosine": 1.0,
            }
        )
    return rows


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = [float(x) for x in values]
    if not vals:
        return float("nan"), float("nan")
    mean = float(statistics.fmean(vals))
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return mean, std


def summarize_hierarchical(anchor_rows):
    """Anchor -> trajectory -> task -> suite-stage macro aggregation.

    Each trajectory contributes exactly one trajectory-level value (the mean of its
    25/50/75 anchors). Each task then averages trajectories equally, and the stage
    summary averages the six Base tasks equally.
    """
    traj_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in anchor_rows:
        key = (str(row["stage"]), int(row["task_id"]), int(row["trajectory_id"]))
        traj_groups.setdefault(key, []).append(row)

    trajectory_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        keys = sorted(k for k in traj_groups if k[0] == stage)
        for _, task_id, trajectory_id in keys:
            rr = traj_groups[(stage, task_id, trajectory_id)]
            out: dict[str, Any] = {
                "stage": stage,
                "task_id": int(task_id),
                "trajectory_id": int(trajectory_id),
                "trajectory_length": int(rr[0]["trajectory_length"]),
                "instruction": str(rr[0]["instruction"]),
                "n_anchors": len(rr),
            }
            for metric in METRICS:
                vals = [float(x[metric]) for x in rr]
                out[f"{metric}_mean"] = float(statistics.fmean(vals))
            trajectory_rows.append(out)

    task_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        task_groups.setdefault((str(row["stage"]), int(row["task_id"])), []).append(row)

    task_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        task_ids = sorted(task for s, task in task_groups if s == stage)
        for task_id in task_ids:
            rr = task_groups[(stage, task_id)]
            out: dict[str, Any] = {
                "stage": stage,
                "task_id": int(task_id),
                "n_trajectories": len(rr),
                "n_anchors": sum(int(x["n_anchors"]) for x in rr),
            }
            for metric in METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in rr]
                mean, std = mean_std(vals)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_trajectory_std"] = std
            task_rows.append(out)

    stage_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        rr = [x for x in task_rows if x["stage"] == stage]
        if not rr:
            continue
        out: dict[str, Any] = {
            "stage": stage,
            "n_tasks": len(rr),
            "n_trajectories_total": sum(int(x["n_trajectories"]) for x in rr),
            "n_anchors_total": sum(int(x["n_anchors"]) for x in rr),
        }
        for metric in METRICS:
            vals = [float(x[f"{metric}_mean"]) for x in rr]
            mean, std = mean_std(vals)
            out[f"{metric}_macro_mean"] = mean
            out[f"{metric}_across_task_std"] = std
        stage_rows.append(out)

    return trajectory_rows, task_rows, stage_rows


def summarize_phases(anchor_rows):
    """Per-phase task means plus suite-level macro phase means."""
    task_phase_groups: dict[tuple[str, int, str, float], list[dict[str, Any]]] = {}
    for row in anchor_rows:
        key = (
            str(row["stage"]),
            int(row["task_id"]),
            str(row["phase"]),
            float(row["phase_fraction"]),
        )
        task_phase_groups.setdefault(key, []).append(row)

    task_phase_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        keys = sorted(k for k in task_phase_groups if k[0] == stage)
        for _, task_id, phase, fraction in keys:
            rr = task_phase_groups[(stage, task_id, phase, fraction)]
            out: dict[str, Any] = {
                "stage": stage,
                "task_id": int(task_id),
                "phase": phase,
                "phase_fraction": float(fraction),
                "n_trajectories": len(rr),
            }
            for metric in METRICS:
                vals = [float(x[metric]) for x in rr]
                mean, std = mean_std(vals)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_trajectory_std"] = std
            task_phase_rows.append(out)

    phase_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in task_phase_rows:
        key = (str(row["stage"]), str(row["phase"]), float(row["phase_fraction"]))
        phase_groups.setdefault(key, []).append(row)

    phase_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        keys = sorted(k for k in phase_groups if k[0] == stage)
        for _, phase, fraction in keys:
            rr = phase_groups[(stage, phase, fraction)]
            out: dict[str, Any] = {
                "stage": stage,
                "phase": phase,
                "phase_fraction": float(fraction),
                "n_tasks": len(rr),
            }
            for metric in METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in rr]
                mean, std = mean_std(vals)
                out[f"{metric}_macro_mean"] = mean
                out[f"{metric}_across_task_std"] = std
            phase_rows.append(out)

    return task_phase_rows, phase_rows


def write_matrix(path: Path, task_rows, task_ids, metric: str) -> None:
    lookup = {
        (str(x["stage"]), int(x["task_id"])): float(x[f"{metric}_mean"])
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
    write_csv(path, rows, ["task", *STAGES])


def main() -> None:
    args = args_parser()
    seed_all(args.seed)
    args.trajectory_fractions = validate_fractions(args.trajectory_fractions)

    if args.split != "all":
        print(
            f"[WARN] split={args.split}. The formal trajectory-balanced protocol uses split=all."
        )
    if args.max_trajectories_per_task > 0:
        print(
            "[WARN] max-trajectories-per-task is active. This is a smoke/debug run, not the formal protocol."
        )

    args.run_root = args.run_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config_yaml = args.config_yaml.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chain = resolve_chain(args.run_root)
    cfg, base_stats = load_base_cfg_stats(
        chain["Base"], canonical_config_yaml=args.config_yaml
    )

    print("=" * 80)
    print("LaWAM Semantic / Conditioning Forgetting -- Trajectory-Balanced Protocol")
    print("=" * 80)
    print("suite                 :", args.suite)
    print("run root              :", args.run_root)
    print("output                :", args.output_dir)
    print("Base tasks            :", args.task_ids)
    print("split                 :", args.split)
    print("trajectory fractions  :", args.trajectory_fractions)
    print("max trajectories/task :", args.max_trajectories_per_task)
    print("batch size            :", args.batch_size)
    print("workers               :", args.num_workers)
    print("device                :", args.device)
    for stage in STAGES:
        print(f"{stage:4s} checkpoint      : {chain[stage]}")
    print("=" * 80)

    meta = {
        "suite": args.suite,
        "run_root": str(args.run_root),
        "output_dir": str(args.output_dir),
        "task_ids": [int(x) for x in args.task_ids],
        "split": args.split,
        "trajectory_fractions": [float(x) for x in args.trajectory_fractions],
        "phase_labels": [phase_label(x) for x in args.trajectory_fractions],
        "max_trajectories_per_task": int(args.max_trajectories_per_task),
        "canonical_config_yaml": str(args.config_yaml),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "checkpoints": {k: str(v) for k, v in chain.items()},
        "protocol": {
            "sampling": (
                "Every filtered Base-task demonstration trajectory contributes one fixed "
                "anchor at each requested progress fraction. Anchor step = round((L-1)*fraction)."
            ),
            "aggregation": (
                "Anchor metrics -> mean within trajectory -> equal-weight mean across trajectories "
                "within task -> equal-weight macro mean across Base tasks."
            ),
            "context": "valid h_VLM tokens strictly before the first latent-action query",
            "conditioning": "h_VLM at latent-action-query positions",
            "relative_l2": "||H_stage-H_base||_F / (||H_base||_F + eps), per anchor",
            "cosine": "mean position-wise cosine similarity, per anchor",
            "query_policy": "each checkpoint uses its own learned act_query and flow_action_query",
            "fixed_input": "processed VLM tensors are materialized once and reused by every checkpoint",
        },
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 1) Materialize the exact trajectory-balanced VLM inputs once.
    # ------------------------------------------------------------------
    collator = build_eval_collator(cfg)
    input_root = args.output_dir / "fixed_inputs"
    task_batches: dict[int, list[Path]] = {}
    task_input_meta: dict[int, dict[str, Any]] = {}

    for task in args.task_ids:
        paths, task_meta = materialize_task_inputs(
            cfg=cfg,
            stats=base_stats,
            collator=collator,
            suite=args.suite,
            task_id=int(task),
            split=args.split,
            fractions=args.trajectory_fractions,
            max_trajectories=int(args.max_trajectories_per_task),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            cache_root=input_root,
            reuse=bool(args.reuse_input_cache),
        )
        task_batches[int(task)] = paths
        task_input_meta[int(task)] = task_meta

    del collator
    gc.collect()

    coverage_rows = []
    for task in args.task_ids:
        tm = task_input_meta[int(task)]
        coverage_rows.append(
            {
                "suite": args.suite,
                "task_id": int(task),
                "available_trajectories": int(tm["available_trajectories"]),
                "selected_trajectories": int(tm["num_trajectories"]),
                "available_steps": int(tm["available_steps"]),
                "anchors_per_trajectory": len(args.trajectory_fractions),
                "total_anchors": int(tm["num_samples"]),
            }
        )
    write_csv(
        args.output_dir / "trajectory_coverage.csv",
        coverage_rows,
        [
            "suite",
            "task_id",
            "available_trajectories",
            "selected_trajectories",
            "available_steps",
            "anchors_per_trajectory",
            "total_anchors",
        ],
    )

    # ------------------------------------------------------------------
    # 2) Load Base once; only VLM runs on the analysis GPU.
    # ------------------------------------------------------------------
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")

    print("\n[model] loading Base LaWAM checkpoint")
    model = LaWAMFramework.from_pretrained(str(chain["Base"]))
    model.eval()
    backend = model.policy_backend
    backend.vlm.to(device)
    backend.vlm.eval()
    num_q = int(backend.num_action_queries)
    print(
        f"[model] VLM on {device}; act queries={num_q}; "
        f"hidden={backend.act_query.shape[-1]}"
    )

    base_act_q = backend.act_query.detach().cpu().float().clone()
    base_flow_q = backend.flow_action_query.detach().cpu().float().clone()

    anchor_rows: list[dict[str, Any]] = []
    query_rows = [
        {
            "suite": args.suite,
            "stage": "Base",
            "act_query_rel_l2": 0.0,
            "act_query_cosine": 1.0,
            "flow_query_rel_l2": 0.0,
            "flow_query_cosine": 1.0,
        }
    ]

    # ------------------------------------------------------------------
    # 3) Base h_VLM cache.
    # ------------------------------------------------------------------
    feature_root = args.output_dir / "base_features"
    if feature_root.exists():
        shutil.rmtree(feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)

    print("\n[Base] extracting Base h_VLM")
    for task in args.task_ids:
        ordinal = 0
        for batch_idx, fixed_path in enumerate(task_batches[int(task)]):
            fixed = torch_load(fixed_path)
            hidden, attn, act_mask = vlm_hidden_only(backend, fixed, device)
            context, context_mask, cond = select_features(
                hidden, attn, act_mask, num_q
            )
            base_path = feature_root / f"task_{task}" / f"batch_{batch_idx:04d}.pt"
            save_base_feature(base_path, context, context_mask, cond, fixed)
            base = torch_load(base_path)
            rr = identity_rows(args.suite, int(task), base, ordinal)
            anchor_rows.extend(rr)
            ordinal += len(rr)

            del fixed, hidden, attn, act_mask, context, context_mask, cond, base
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(
            f"[Base] task={task}: trajectories={task_input_meta[int(task)]['num_trajectories']} "
            f"anchors={ordinal}"
        )

    # ------------------------------------------------------------------
    # 4) CL1-CL4: same exact anchors, each checkpoint's own VLM + queries.
    # ------------------------------------------------------------------
    for stage in STAGES[1:]:
        print(f"\n[{stage}] loading VLM + learned queries")
        load_vlm_queries(model, chain[stage])
        backend.vlm.eval()

        cur_act_q = backend.act_query.detach().cpu().float()
        cur_flow_q = backend.flow_action_query.detach().cpu().float()
        query_row = {
            "suite": args.suite,
            "stage": stage,
            "act_query_rel_l2": rel_l2(cur_act_q, base_act_q),
            "act_query_cosine": token_cos(cur_act_q, base_act_q),
            "flow_query_rel_l2": rel_l2(cur_flow_q, base_flow_q),
            "flow_query_cosine": token_cos(cur_flow_q, base_flow_q),
        }
        query_rows.append(query_row)
        print(
            f"[{stage}] act_query: rel_l2={query_row['act_query_rel_l2']:.6f}, "
            f"cos={query_row['act_query_cosine']:.6f}"
        )

        for task in args.task_ids:
            ordinal = 0
            for batch_idx, fixed_path in enumerate(task_batches[int(task)]):
                fixed = torch_load(fixed_path)
                base = torch_load(
                    feature_root / f"task_{task}" / f"batch_{batch_idx:04d}.pt"
                )
                hidden, attn, act_mask = vlm_hidden_only(backend, fixed, device)
                context, context_mask, cond = select_features(
                    hidden, attn, act_mask, num_q
                )
                rr = compare_batch(
                    args.suite,
                    int(task),
                    stage,
                    base,
                    context,
                    context_mask,
                    cond,
                    ordinal,
                )
                anchor_rows.extend(rr)
                ordinal += len(rr)

                del fixed, base, hidden, attn, act_mask, context, context_mask, cond
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            print(
                f"[{stage}] task={task}: trajectories={task_input_meta[int(task)]['num_trajectories']} "
                f"anchors={ordinal}"
            )

    # ------------------------------------------------------------------
    # 5) Hierarchical summaries and matrices.
    # ------------------------------------------------------------------
    anchor_fields = [
        "suite",
        "task_id",
        "stage",
        "anchor_ordinal",
        "trajectory_id",
        "trajectory_length",
        "trajectory_step",
        "phase",
        "phase_fraction",
        "instruction",
        "context_tokens",
        "conditioning_tokens",
        *METRICS,
    ]
    write_csv(
        args.output_dir / "semantic_conditioning_anchor_metrics.csv",
        anchor_rows,
        anchor_fields,
    )

    trajectory_rows, task_rows, stage_rows = summarize_hierarchical(anchor_rows)
    task_phase_rows, phase_rows = summarize_phases(anchor_rows)

    trajectory_fields = [
        "stage",
        "task_id",
        "trajectory_id",
        "trajectory_length",
        "instruction",
        "n_anchors",
        *[f"{metric}_mean" for metric in METRICS],
    ]
    write_csv(
        args.output_dir / "semantic_conditioning_trajectory_summary.csv",
        trajectory_rows,
        trajectory_fields,
    )

    task_fields = ["stage", "task_id", "n_trajectories", "n_anchors"]
    for metric in METRICS:
        task_fields += [f"{metric}_mean", f"{metric}_trajectory_std"]
    write_csv(
        args.output_dir / "semantic_conditioning_task_summary.csv",
        task_rows,
        task_fields,
    )

    stage_fields = [
        "stage",
        "n_tasks",
        "n_trajectories_total",
        "n_anchors_total",
    ]
    for metric in METRICS:
        stage_fields += [f"{metric}_macro_mean", f"{metric}_across_task_std"]
    write_csv(
        args.output_dir / "semantic_conditioning_stage_summary.csv",
        stage_rows,
        stage_fields,
    )

    task_phase_fields = [
        "stage",
        "task_id",
        "phase",
        "phase_fraction",
        "n_trajectories",
    ]
    for metric in METRICS:
        task_phase_fields += [f"{metric}_mean", f"{metric}_trajectory_std"]
    write_csv(
        args.output_dir / "semantic_conditioning_phase_task_summary.csv",
        task_phase_rows,
        task_phase_fields,
    )

    phase_fields = ["stage", "phase", "phase_fraction", "n_tasks"]
    for metric in METRICS:
        phase_fields += [f"{metric}_macro_mean", f"{metric}_across_task_std"]
    write_csv(
        args.output_dir / "semantic_conditioning_phase_summary.csv",
        phase_rows,
        phase_fields,
    )

    write_csv(
        args.output_dir / "query_drift.csv",
        query_rows,
        [
            "suite",
            "stage",
            "act_query_rel_l2",
            "act_query_cosine",
            "flow_query_rel_l2",
            "flow_query_cosine",
        ],
    )

    for metric in METRICS:
        write_matrix(
            args.output_dir / f"matrix_{metric}.csv",
            task_rows,
            args.task_ids,
            metric,
        )

    print("\n" + "=" * 80)
    print("Trajectory-balanced analysis complete")
    print("=" * 80)
    print("coverage        :", args.output_dir / "trajectory_coverage.csv")
    print("anchor metrics  :", args.output_dir / "semantic_conditioning_anchor_metrics.csv")
    print("trajectory      :", args.output_dir / "semantic_conditioning_trajectory_summary.csv")
    print("task summary    :", args.output_dir / "semantic_conditioning_task_summary.csv")
    print("stage summary   :", args.output_dir / "semantic_conditioning_stage_summary.csv")
    print("phase summary   :", args.output_dir / "semantic_conditioning_phase_summary.csv")
    print("query drift     :", args.output_dir / "query_drift.csv")
    for metric in METRICS:
        print("matrix          :", args.output_dir / f"matrix_{metric}.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()