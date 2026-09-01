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

# H_act-only analysis. We still feed the complete LaWAM VLM input, but we only
# retain the latent-action-query hidden states.
VLM_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "act_placeholder_mask",
    "flow_placeholder_mask",
)

RAW_SAMPLE_KEYS = (
    "primary_videos",
    "wrist_images",
    "state",
    "action",
    "lang",
    "embodiment_id",
    "action_hz",
)


# -----------------------------------------------------------------------------
# CLI / configuration
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LaWAM H_act instruction-contrast analysis for semantic forgetting. "
            "For each Base-task observation anchor, compare the correct instruction "
            "against mismatched CL-task instructions (default tasks 6-9) within "
            "Base/CL1/CL2/CL3/CL4 checkpoints."
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
    p.add_argument(
        "--base-task-ids",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Old/Base tasks whose observations are used.",
    )
    p.add_argument(
        "--negative-task-ids",
        nargs="+",
        type=int,
        default=[6, 7, 8, 9],
        help=(
            "Mismatched instruction tasks. In the current Seq-FT protocol these are "
            "CL1/CL2/CL3/CL4 tasks respectively."
        ),
    )
    p.add_argument(
        "--trajectory-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="Trajectory progress anchors. Formal protocol: 0.25 0.50 0.75.",
    )
    p.add_argument(
        "--max-trajectories-per-task",
        type=int,
        default=0,
        help="0 = all trajectories. Positive values are smoke/debug only.",
    )
    p.add_argument(
        "--anchors-per-batch",
        type=int,
        default=1,
        help=(
            "Number of observation anchors collated at once. Actual VLM batch size is "
            "anchors_per_batch * (1 + number_of_negative_tasks). Default 1 is safest."
        ),
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Formal protocol should use all.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--reuse-anchor-cache",
        action="store_true",
        help="Reuse deterministic raw anchor cache if metadata matches.",
    )
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
    rd = run_dir(base_ckpt)
    canonical_config_yaml = canonical_config_yaml.expanduser().resolve()
    if not canonical_config_yaml.is_file():
        raise FileNotFoundError(f"Canonical config not found: {canonical_config_yaml}")

    canonical_cfg = OmegaConf.load(canonical_config_yaml)
    run_cfg = OmegaConf.load(rd / "config.yaml")
    cfg = OmegaConf.merge(canonical_cfg, run_cfg)

    required = (
        "data_root_dir",
        "data_mix",
        "image_resolution",
        "num_frames",
        "sec_chunk",
    )
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


def force_eval_transforms(mixture) -> None:
    for single_dataset in getattr(mixture, "datasets", []):
        transforms = getattr(single_dataset, "transforms", None)
        if transforms is not None and hasattr(transforms, "eval"):
            transforms.eval()


def worker_init(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Trajectory-balanced anchor sampling
# -----------------------------------------------------------------------------

def choose_trajectory_indices(total: int, max_trajectories: int) -> list[int]:
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

    selected = choose_trajectory_indices(len(traj_ids), int(max_trajectories))
    specs: list[dict[str, Any]] = []
    warnings: list[str] = []

    for traj_local_idx in selected:
        trajectory_id = int(traj_ids[traj_local_idx])
        length = int(traj_lengths[traj_local_idx])
        if length <= 0:
            warnings.append(
                f"trajectory {trajectory_id} has non-positive length={length}; skipped"
            )
            continue
        steps = [int(round((length - 1) * float(frac))) for frac in fractions]
        if len(set(steps)) != len(steps):
            warnings.append(
                f"trajectory {trajectory_id} length={length} maps multiple phases "
                f"to the same step: {steps}"
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
    return specs, selected, warnings


class TrajectoryAnchorDataset(Dataset):
    """Exact trajectory anchors using the same LaWAM dataset transforms/output path."""

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


def slim_raw_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in RAW_SAMPLE_KEYS:
        if key not in sample:
            raise KeyError(f"Raw LaWAM sample missing required key {key!r}")
        value = sample[key]
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().contiguous()
        else:
            out[key] = value
    return out


class RawAnchorCollator:
    def __call__(self, items):
        specs, samples = zip(*items)
        return {
            "specs": [dict(x) for x in specs],
            "samples": [slim_raw_sample(dict(x)) for x in samples],
        }


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_anchor_manifest(
    path: Path,
    suite: str,
    task_id: int,
    specs: Sequence[dict[str, Any]],
    positive_instruction: str,
) -> None:
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
                "positive_instruction": positive_instruction,
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
            "positive_instruction",
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
    anchors_per_batch: int,
) -> bool:
    return (
        meta.get("suite") == suite
        and int(meta.get("task_id", -1)) == int(task_id)
        and meta.get("split") == split
        and [float(x) for x in meta.get("trajectory_fractions", [])]
        == [float(x) for x in fractions]
        and int(meta.get("max_trajectories_per_task", -999))
        == int(max_trajectories)
        and int(meta.get("anchors_per_batch", -1)) == int(anchors_per_batch)
        and int(meta.get("num_batches", 0)) > 0
    )


def load_raw_cache(path: Path):
    # Raw samples are plain Python dict/list objects plus tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


def materialize_raw_anchor_cache(
    *,
    cfg,
    stats,
    suite: str,
    task_id: int,
    split: str,
    fractions: Sequence[float],
    max_trajectories: int,
    anchors_per_batch: int,
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
            anchors_per_batch=anchors_per_batch,
        ) and len(paths) == int(meta.get("num_batches", 0)):
            print(
                f"[cache] task={task_id}: reuse trajectories={meta['num_trajectories']} "
                f"anchors={meta['num_anchors']}"
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
    force_eval_transforms(mixture)

    datasets = list(getattr(mixture, "datasets", []))
    if len(datasets) != 1:
        raise RuntimeError(
            f"Expected one filtered LIBERO dataset, got {len(datasets)} "
            f"for {suite} task={task_id}"
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
        "batch_size": int(anchors_per_batch),
        "shuffle": False,
        "drop_last": False,
        "collate_fn": RawAnchorCollator(),
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
    langs_seen: set[str] = set()
    for batch_idx, raw_batch in enumerate(loader):
        for sample in raw_batch["samples"]:
            langs_seen.add(str(sample["lang"]))
        path = td / f"batch_{batch_idx:04d}.pt"
        torch.save(raw_batch, path)
        paths.append(path)
        count += len(raw_batch["samples"])

    expected = len(selected_traj_indices) * len(fractions)
    if count != expected:
        raise RuntimeError(
            f"Anchor count mismatch for task={task_id}: wrote={count}, expected={expected}"
        )
    if len(langs_seen) != 1:
        raise RuntimeError(
            f"Expected exactly one instruction for filtered {suite} task={task_id}, "
            f"got {sorted(langs_seen)}"
        )
    positive_instruction = next(iter(langs_seen))

    meta = {
        "suite": suite,
        "task_id": int(task_id),
        "split": split,
        "trajectory_fractions": [float(x) for x in fractions],
        "phase_labels": [phase_label(x) for x in fractions],
        "max_trajectories_per_task": int(max_trajectories),
        "anchors_per_batch": int(anchors_per_batch),
        "available_trajectories": int(len(single.trajectory_ids)),
        "available_steps": int(len(single)),
        "selected_trajectory_local_indices": [int(x) for x in selected_traj_indices],
        "selected_trajectory_ids": sorted({int(x["trajectory_id"]) for x in specs}),
        "num_trajectories": int(len(selected_traj_indices)),
        "num_anchors": int(count),
        "num_batches": int(len(paths)),
        "positive_instruction": positive_instruction,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_anchor_manifest(
        td / "anchor_manifest.csv",
        suite,
        task_id,
        specs,
        positive_instruction,
    )

    print(
        f"[anchors] suite={suite} task={task_id} split={split} "
        f"trajectories={meta['num_trajectories']}/{meta['available_trajectories']} "
        f"fractions={meta['phase_labels']} anchors={count}"
    )
    print(f"[instruction] task={task_id}: {positive_instruction}")
    print(f"[cache] task={task_id}: wrote {len(paths)} raw-anchor batches")
    return paths, meta


# -----------------------------------------------------------------------------
# Instruction discovery for CL1-CL4 tasks
# -----------------------------------------------------------------------------

def sample_from_exact_position(mixture, trajectory_id: int, trajectory_step: int):
    datasets = list(getattr(mixture, "datasets", []))
    if len(datasets) != 1:
        raise RuntimeError(f"Expected one underlying dataset, got {len(datasets)}")
    single = datasets[0]
    selected_video_keys = mixture._select_video_keys_for_sample(single, 0)
    if not selected_video_keys:
        raise ValueError("No video keys resolved while discovering instruction")
    raw_data = single.get_step_data(
        int(trajectory_id),
        int(trajectory_step),
        modality_keys_override={"video": selected_video_keys},
    )
    transforms = mixture._get_transforms_for_selected_video_keys(
        single, selected_video_keys
    )
    data = transforms(raw_data)
    return mixture._build_output_sample(single, data, selected_video_keys)


def discover_task_instruction(cfg, stats, suite: str, task_id: int, split: str, seed: int) -> str:
    mixture = get_vla_dataset(
        data_cfg=task_data_cfg(cfg, suite, task_id),
        mode=split,
        balance_dataset_weights=False,
        seed=seed,
        framework_name=str(cfg.framework.name),
        dataset_statistics_override=stats,
    )
    force_eval_transforms(mixture)
    datasets = list(getattr(mixture, "datasets", []))
    if len(datasets) != 1:
        raise RuntimeError(
            f"Expected one filtered dataset while discovering {suite} task={task_id}, "
            f"got {len(datasets)}"
        )
    single = datasets[0]
    if len(single.trajectory_ids) == 0:
        raise RuntimeError(f"No trajectories for {suite} task={task_id}")
    traj_id = int(single.trajectory_ids[0])
    length = int(single.trajectory_lengths[0])
    step = max(0, (length - 1) // 2)
    sample = sample_from_exact_position(mixture, traj_id, step)
    instruction = str(sample["lang"])
    if not instruction:
        raise RuntimeError(f"Empty instruction for {suite} task={task_id}")
    print(f"[instruction] task={task_id}: {instruction}")
    del sample, mixture
    gc.collect()
    return instruction


# -----------------------------------------------------------------------------
# Build positive + mismatched-instruction VLM batches
# -----------------------------------------------------------------------------

def build_instruction_variant_batch(
    collator: LatentWorldTrainCollator,
    raw_batch: dict[str, Any],
    *,
    base_task_id: int,
    negative_instruction_map: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variants: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    negative_ids = list(negative_instruction_map.keys())

    for anchor_local_idx, (spec, sample) in enumerate(
        zip(raw_batch["specs"], raw_batch["samples"])
    ):
        positive_instruction = str(sample["lang"])
        instruction_conditions = [(int(base_task_id), "positive", positive_instruction)]
        instruction_conditions += [
            (int(neg_id), "negative", str(negative_instruction_map[int(neg_id)]))
            for neg_id in negative_ids
        ]

        for instruction_task_id, role, instruction in instruction_conditions:
            variant = dict(sample)  # shallow copy; tensors are intentionally shared/read-only
            variant["lang"] = instruction
            variants.append(variant)
            metadata.append(
                {
                    "base_task_id": int(base_task_id),
                    "anchor_local_idx": int(anchor_local_idx),
                    "trajectory_local_index": int(spec["trajectory_local_index"]),
                    "trajectory_id": int(spec["trajectory_id"]),
                    "trajectory_length": int(spec["trajectory_length"]),
                    "trajectory_step": int(spec["trajectory_step"]),
                    "phase": str(spec["phase"]),
                    "phase_fraction": float(spec["phase_fraction"]),
                    "instruction_task_id": int(instruction_task_id),
                    "instruction_role": role,
                    "instruction": instruction,
                    "positive_instruction": positive_instruction,
                }
            )

    batch = collator(variants)
    expected = len(raw_batch["samples"]) * (1 + len(negative_ids))
    if int(batch["input_ids"].shape[0]) != expected:
        raise RuntimeError(
            f"Variant batch size mismatch: got={batch['input_ids'].shape[0]} expected={expected}"
        )
    return batch, metadata


# -----------------------------------------------------------------------------
# Checkpoint loading and H_act extraction
# -----------------------------------------------------------------------------

def torch_load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(state):
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_vlm_queries(model, ckpt: Path) -> None:
    """Swap only VLM + learned queries, the only checkpoint parts that affect H_act."""
    state = normalize_state_dict(torch_load_state(ckpt))
    backend = model.policy_backend
    prefix = "policy_backend.vlm."
    vlm_state = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
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


def move_vlm_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in VLM_KEYS:
        value = batch[key]
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def extract_hact_only(backend, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Run LaWAM's VLM path only up to H_act; do not use VLMToLAM/LaWM/Flow."""
    batch = move_vlm_batch(batch, device)
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

        vlm_out = backend.vlm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
        )
        hidden = getattr(vlm_out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("VLM output has no last_hidden_state")

        act_mask = batch["act_placeholder_mask"].to(device=hidden.device, dtype=torch.bool)
        bsz, _, hidden_dim = hidden.shape
        q = int(backend.num_action_queries)
        counts = act_mask.sum(dim=1)
        if not torch.all(counts == q):
            raise RuntimeError(
                f"act query count mismatch: {counts.detach().cpu().tolist()} expected={q}"
            )
        hact = hidden[act_mask].view(int(bsz), q, int(hidden_dim))

    return hact.detach().cpu().to(torch.bfloat16)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def relative_l2(negative: torch.Tensor, positive: torch.Tensor, eps: float = 1e-8) -> float:
    negative = negative.float()
    positive = positive.float()
    num = torch.linalg.vector_norm(negative - positive)
    den = torch.linalg.vector_norm(positive).clamp_min(eps)
    return float((num / den).item())


def token_cosine(negative: torch.Tensor, positive: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(negative.float(), positive.float(), dim=-1, eps=1e-8)
        .mean()
        .item()
    )


def safe_ratio(value: float, base_value: float, eps: float = 1e-6) -> float:
    return float(value / max(float(base_value), eps))


def stage_number(stage: str) -> int:
    if stage == "Base":
        return 0
    if not stage.startswith("CL"):
        raise ValueError(stage)
    return int(stage[2:])


def current_task_for_stage(stage: str) -> int | None:
    n = stage_number(stage)
    return None if n == 0 else 5 + n


def negative_status(stage: str, negative_task_id: int) -> str:
    current = current_task_for_stage(stage)
    if current is None:
        return "base_unseen"
    if int(negative_task_id) == int(current):
        return "current"
    if int(negative_task_id) < int(current):
        return "seen_prior"
    return "unseen_future"


def anchor_reference_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["base_task_id"]),
        int(row["trajectory_id"]),
        str(row["phase"]),
        int(row["negative_task_id"]),
    )


def compute_contrast_rows_for_batch(
    *,
    suite: str,
    stage: str,
    hact: torch.Tensor,
    variant_meta: list[dict[str, Any]],
    negative_task_ids: Sequence[int],
    base_reference: dict[tuple[Any, ...], dict[str, float]],
    anchor_ordinal_offset: int,
) -> list[dict[str, Any]]:
    if int(hact.shape[0]) != len(variant_meta):
        raise RuntimeError(
            f"H_act / metadata size mismatch: {hact.shape[0]} vs {len(variant_meta)}"
        )

    groups: dict[int, list[int]] = {}
    for idx, meta in enumerate(variant_meta):
        groups.setdefault(int(meta["anchor_local_idx"]), []).append(idx)

    rows: list[dict[str, Any]] = []
    for local_anchor_idx in sorted(groups):
        idxs = groups[local_anchor_idx]
        pos_candidates = [i for i in idxs if variant_meta[i]["instruction_role"] == "positive"]
        if len(pos_candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one positive variant for anchor {local_anchor_idx}, "
                f"got {len(pos_candidates)}"
            )
        pos_idx = pos_candidates[0]
        pos_h = hact[pos_idx]
        pos_meta = variant_meta[pos_idx]

        neg_by_task = {
            int(variant_meta[i]["instruction_task_id"]): i
            for i in idxs
            if variant_meta[i]["instruction_role"] == "negative"
        }
        missing_neg = [int(x) for x in negative_task_ids if int(x) not in neg_by_task]
        if missing_neg:
            raise RuntimeError(
                f"Missing negative variants {missing_neg} for anchor {local_anchor_idx}"
            )

        for neg_task in negative_task_ids:
            neg_idx = neg_by_task[int(neg_task)]
            neg_h = hact[neg_idx]
            neg_meta = variant_meta[neg_idx]

            rel = relative_l2(neg_h, pos_h)
            cos = token_cosine(neg_h, pos_h)
            gap = float(1.0 - cos)

            row = {
                "suite": suite,
                "base_task_id": int(pos_meta["base_task_id"]),
                "stage": stage,
                "anchor_ordinal": int(anchor_ordinal_offset + local_anchor_idx),
                "trajectory_id": int(pos_meta["trajectory_id"]),
                "trajectory_length": int(pos_meta["trajectory_length"]),
                "trajectory_step": int(pos_meta["trajectory_step"]),
                "phase": str(pos_meta["phase"]),
                "phase_fraction": float(pos_meta["phase_fraction"]),
                "positive_instruction": str(pos_meta["positive_instruction"]),
                "negative_task_id": int(neg_task),
                "negative_instruction": str(neg_meta["instruction"]),
                "negative_status": negative_status(stage, int(neg_task)),
                "hact_rel_l2": rel,
                "hact_cosine": cos,
                "semantic_gap": gap,
            }

            key = anchor_reference_key(row)
            if stage == "Base":
                base_reference[key] = {
                    "hact_rel_l2": rel,
                    "semantic_gap": gap,
                }
                row["l2_retention"] = 1.0
                row["semantic_retention"] = 1.0
            else:
                if key not in base_reference:
                    raise KeyError(f"Missing Base reference for {key}")
                ref = base_reference[key]
                row["l2_retention"] = safe_ratio(rel, ref["hact_rel_l2"])
                row["semantic_retention"] = safe_ratio(gap, ref["semantic_gap"])

            rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# Hierarchical summaries
# -----------------------------------------------------------------------------

def mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = [float(x) for x in values]
    if not vals:
        return float("nan"), float("nan")
    mean = float(statistics.fmean(vals))
    std = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return mean, std


SUMMARY_METRICS = (
    "hact_rel_l2",
    "hact_cosine",
    "semantic_gap",
    "l2_retention",
    "semantic_retention",
)


def summarize_hierarchical(anchor_rows):
    # Anchor (p25/p50/p75) -> trajectory, separately for each negative instruction.
    traj_groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for row in anchor_rows:
        key = (
            str(row["stage"]),
            int(row["base_task_id"]),
            int(row["trajectory_id"]),
            int(row["negative_task_id"]),
        )
        traj_groups.setdefault(key, []).append(row)

    trajectory_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        keys = sorted(k for k in traj_groups if k[0] == stage)
        for _, base_task_id, trajectory_id, neg_task in keys:
            rr = traj_groups[(stage, base_task_id, trajectory_id, neg_task)]
            out: dict[str, Any] = {
                "stage": stage,
                "base_task_id": int(base_task_id),
                "trajectory_id": int(trajectory_id),
                "trajectory_length": int(rr[0]["trajectory_length"]),
                "positive_instruction": str(rr[0]["positive_instruction"]),
                "negative_task_id": int(neg_task),
                "negative_instruction": str(rr[0]["negative_instruction"]),
                "negative_status": str(rr[0]["negative_status"]),
                "n_anchors": len(rr),
            }
            for metric in SUMMARY_METRICS:
                out[f"{metric}_mean"] = float(
                    statistics.fmean(float(x[metric]) for x in rr)
                )
            trajectory_rows.append(out)

    # Trajectory -> Base task, still separately for each negative instruction.
    task_neg_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in trajectory_rows:
        key = (
            str(row["stage"]),
            int(row["base_task_id"]),
            int(row["negative_task_id"]),
        )
        task_neg_groups.setdefault(key, []).append(row)

    task_negative_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        keys = sorted(k for k in task_neg_groups if k[0] == stage)
        for _, base_task_id, neg_task in keys:
            rr = task_neg_groups[(stage, base_task_id, neg_task)]
            out: dict[str, Any] = {
                "stage": stage,
                "base_task_id": int(base_task_id),
                "positive_instruction": str(rr[0]["positive_instruction"]),
                "negative_task_id": int(neg_task),
                "negative_instruction": str(rr[0]["negative_instruction"]),
                "negative_status": str(rr[0]["negative_status"]),
                "n_trajectories": len(rr),
                "n_anchors": sum(int(x["n_anchors"]) for x in rr),
            }
            for metric in SUMMARY_METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in rr]
                mean, std = mean_std(vals)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_trajectory_std"] = std
            task_negative_rows.append(out)

    # Base task -> suite macro, separately for each negative task.
    stage_neg_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in task_negative_rows:
        stage_neg_groups.setdefault(
            (str(row["stage"]), int(row["negative_task_id"])), []
        ).append(row)

    stage_negative_rows: list[dict[str, Any]] = []
    for stage in STAGES:
        negs = sorted(neg for s, neg in stage_neg_groups if s == stage)
        for neg_task in negs:
            rr = stage_neg_groups[(stage, neg_task)]
            out: dict[str, Any] = {
                "stage": stage,
                "negative_task_id": int(neg_task),
                "negative_instruction": str(rr[0]["negative_instruction"]),
                "negative_status": negative_status(stage, int(neg_task)),
                "n_base_tasks": len(rr),
            }
            for metric in SUMMARY_METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in rr]
                mean, std = mean_std(vals)
                out[f"{metric}_macro_mean"] = mean
                out[f"{metric}_across_task_std"] = std
            stage_negative_rows.append(out)

    return trajectory_rows, task_negative_rows, stage_negative_rows


def summarize_task_all_negative(task_negative_rows):
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in task_negative_rows:
        groups.setdefault((str(row["stage"]), int(row["base_task_id"])), []).append(row)

    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        tasks = sorted(task for s, task in groups if s == stage)
        for task in tasks:
            rr = groups[(stage, task)]
            out: dict[str, Any] = {
                "stage": stage,
                "base_task_id": int(task),
                "n_negative_instructions": len(rr),
            }
            for metric in SUMMARY_METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in rr]
                out[f"all_negative_{metric}_mean"] = float(statistics.fmean(vals))
            rows.append(out)
    return rows


def summarize_stage(task_negative_rows):
    """Suite-level summaries: all negatives, current task, seen-prior, unseen-future."""
    rows: list[dict[str, Any]] = []
    for stage in STAGES:
        stage_rows = [x for x in task_negative_rows if x["stage"] == stage]
        if not stage_rows:
            continue

        out: dict[str, Any] = {"stage": stage}
        current = current_task_for_stage(stage)
        out["current_negative_task_id"] = "" if current is None else int(current)

        for category, predicate in (
            ("all", lambda x: True),
            ("current", lambda x: x["negative_status"] == "current"),
            ("seen_prior", lambda x: x["negative_status"] == "seen_prior"),
            (
                "unseen",
                lambda x: x["negative_status"] in {"base_unseen", "unseen_future"},
            ),
        ):
            selected = [x for x in stage_rows if predicate(x)]
            out[f"{category}_n_task_negative_pairs"] = len(selected)
            for metric in SUMMARY_METRICS:
                vals = [float(x[f"{metric}_mean"]) for x in selected]
                out[f"{category}_{metric}"] = (
                    float(statistics.fmean(vals)) if vals else float("nan")
                )
        rows.append(out)
    return rows


def write_task_matrix(path: Path, task_rows, base_task_ids, value_key: str) -> None:
    lookup = {
        (str(x["stage"]), int(x["base_task_id"])): float(x[value_key])
        for x in task_rows
    }
    rows = []
    for task in base_task_ids:
        row = {"task": f"task_{task}"}
        for stage in STAGES:
            row[stage] = lookup.get((stage, int(task)), float("nan"))
        rows.append(row)

    macro = {"task": "macro_mean"}
    for stage in STAGES:
        vals = [
            lookup[(stage, int(task))]
            for task in base_task_ids
            if (stage, int(task)) in lookup
        ]
        macro[stage] = statistics.fmean(vals) if vals else float("nan")
    rows.append(macro)
    write_csv(path, rows, ["task", *STAGES])


def current_negative_task_rows(task_negative_rows):
    rows: list[dict[str, Any]] = []
    for stage in STAGES[1:]:
        current = current_task_for_stage(stage)
        assert current is not None
        selected = [
            x
            for x in task_negative_rows
            if x["stage"] == stage and int(x["negative_task_id"]) == int(current)
        ]
        rows.extend(selected)
    return rows


def write_current_matrix(path: Path, task_negative_rows, base_task_ids, metric: str) -> None:
    current_rows = current_negative_task_rows(task_negative_rows)
    lookup = {
        (str(x["stage"]), int(x["base_task_id"])): float(x[f"{metric}_mean"])
        for x in current_rows
    }
    rows = []
    for task in base_task_ids:
        row = {"task": f"task_{task}", "Base": float("nan")}
        for stage in STAGES[1:]:
            row[stage] = lookup.get((stage, int(task)), float("nan"))
        rows.append(row)
    macro = {"task": "macro_mean", "Base": float("nan")}
    for stage in STAGES[1:]:
        vals = [
            lookup[(stage, int(task))]
            for task in base_task_ids
            if (stage, int(task)) in lookup
        ]
        macro[stage] = statistics.fmean(vals) if vals else float("nan")
    rows.append(macro)
    write_csv(path, rows, ["task", *STAGES])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    args.trajectory_fractions = validate_fractions(args.trajectory_fractions)

    if args.anchors_per_batch <= 0:
        raise ValueError("--anchors-per-batch must be > 0")
    if len(set(args.base_task_ids)) != len(args.base_task_ids):
        raise ValueError("--base-task-ids must be unique")
    if len(set(args.negative_task_ids)) != len(args.negative_task_ids):
        raise ValueError("--negative-task-ids must be unique")
    overlap = set(args.base_task_ids) & set(args.negative_task_ids)
    if overlap:
        raise ValueError(f"Base/negative task ids overlap: {sorted(overlap)}")
    if args.split != "all":
        print(f"[WARN] split={args.split}; formal protocol uses split=all")
    if args.max_trajectories_per_task > 0:
        print(
            "[WARN] max-trajectories-per-task is active: smoke/debug run, not formal protocol"
        )

    args.run_root = args.run_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config_yaml = args.config_yaml.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chain = resolve_chain(args.run_root)
    cfg, base_stats = load_base_cfg_stats(
        chain["Base"], canonical_config_yaml=args.config_yaml
    )

    print("=" * 88)
    print("LaWAM H_act Semantic Forgetting -- Instruction Contrast")
    print("=" * 88)
    print("suite                  :", args.suite)
    print("run root               :", args.run_root)
    print("output                 :", args.output_dir)
    print("Base observation tasks :", args.base_task_ids)
    print("negative task ids      :", args.negative_task_ids)
    print("split                  :", args.split)
    print("trajectory fractions   :", args.trajectory_fractions)
    print("max trajectories/task  :", args.max_trajectories_per_task)
    print("anchors / raw batch    :", args.anchors_per_batch)
    print(
        "actual VLM batch size   :",
        args.anchors_per_batch * (1 + len(args.negative_task_ids)),
    )
    print("workers                :", args.num_workers)
    print("device                 :", args.device)
    for stage in STAGES:
        print(f"{stage:4s} checkpoint       : {chain[stage]}")
    print("=" * 88)

    # ------------------------------------------------------------------
    # 1) Discover mismatched instructions from the actual local filtered data.
    # ------------------------------------------------------------------
    print("\n[instructions] discovering CL-task instructions")
    negative_instruction_map: dict[int, str] = {}
    for task_id in args.negative_task_ids:
        negative_instruction_map[int(task_id)] = discover_task_instruction(
            cfg,
            base_stats,
            args.suite,
            int(task_id),
            args.split,
            int(args.seed),
        )

    instruction_rows = [
        {
            "suite": args.suite,
            "task_id": int(task_id),
            "role": "negative_mismatched",
            "instruction": negative_instruction_map[int(task_id)],
        }
        for task_id in args.negative_task_ids
    ]

    # ------------------------------------------------------------------
    # 2) Materialize every Base-task trajectory anchor once as a raw LaWAM sample.
    #    This avoids re-decoding videos for every checkpoint while allowing us to
    #    rebuild the language portion for positive / negative instructions.
    # ------------------------------------------------------------------
    cache_root = args.output_dir / "raw_anchor_cache"
    task_batches: dict[int, list[Path]] = {}
    task_meta: dict[int, dict[str, Any]] = {}

    for task_id in args.base_task_ids:
        paths, tm = materialize_raw_anchor_cache(
            cfg=cfg,
            stats=base_stats,
            suite=args.suite,
            task_id=int(task_id),
            split=args.split,
            fractions=args.trajectory_fractions,
            max_trajectories=int(args.max_trajectories_per_task),
            anchors_per_batch=int(args.anchors_per_batch),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            cache_root=cache_root,
            reuse=bool(args.reuse_anchor_cache),
        )
        task_batches[int(task_id)] = paths
        task_meta[int(task_id)] = tm
        instruction_rows.append(
            {
                "suite": args.suite,
                "task_id": int(task_id),
                "role": "positive_base_task",
                "instruction": str(tm["positive_instruction"]),
            }
        )

    write_csv(
        args.output_dir / "instruction_map.csv",
        instruction_rows,
        ["suite", "task_id", "role", "instruction"],
    )

    coverage_rows = []
    for task_id in args.base_task_ids:
        tm = task_meta[int(task_id)]
        coverage_rows.append(
            {
                "suite": args.suite,
                "base_task_id": int(task_id),
                "available_trajectories": int(tm["available_trajectories"]),
                "selected_trajectories": int(tm["num_trajectories"]),
                "available_steps": int(tm["available_steps"]),
                "anchors_per_trajectory": len(args.trajectory_fractions),
                "total_anchors": int(tm["num_anchors"]),
                "positive_instruction": str(tm["positive_instruction"]),
            }
        )
    write_csv(
        args.output_dir / "trajectory_coverage.csv",
        coverage_rows,
        [
            "suite",
            "base_task_id",
            "available_trajectories",
            "selected_trajectories",
            "available_steps",
            "anchors_per_trajectory",
            "total_anchors",
            "positive_instruction",
        ],
    )

    meta = {
        "suite": args.suite,
        "run_root": str(args.run_root),
        "output_dir": str(args.output_dir),
        "base_task_ids": [int(x) for x in args.base_task_ids],
        "negative_task_ids": [int(x) for x in args.negative_task_ids],
        "negative_instruction_map": {
            str(k): v for k, v in negative_instruction_map.items()
        },
        "split": args.split,
        "trajectory_fractions": [float(x) for x in args.trajectory_fractions],
        "max_trajectories_per_task": int(args.max_trajectories_per_task),
        "anchors_per_batch": int(args.anchors_per_batch),
        "actual_vlm_batch_size": int(
            args.anchors_per_batch * (1 + len(args.negative_task_ids))
        ),
        "canonical_config_yaml": str(args.config_yaml),
        "checkpoints": {k: str(v) for k, v in chain.items()},
        "protocol": {
            "observation": (
                "Base-task observations only. Each demonstration trajectory contributes "
                "anchors at the requested progress fractions."
            ),
            "instruction_conditions": (
                "For every observation anchor, compare its correct Base-task instruction "
                "with instructions from CL tasks 6-9 (mismatched/counterfactual task instructions)."
            ),
            "feature": "H_act only: h_VLM at the eight latent-action-query positions.",
            "relative_l2": "||H_neg-H_pos||_F / (||H_pos||_F + eps), within checkpoint.",
            "cosine": "mean query-wise cosine(H_neg_q, H_pos_q), within checkpoint.",
            "semantic_gap": "1 - cosine; larger means stronger instruction separation.",
            "semantic_retention": (
                "semantic_gap(stage) / semantic_gap(Base) for the same observation anchor "
                "and same negative instruction pair. Values <1 indicate reduced separation."
            ),
            "aggregation": (
                "p25/p50/p75 anchors -> equal-weight trajectory mean -> equal-weight "
                "trajectory mean within Base task -> equal-weight Base-task macro mean."
            ),
            "checkpoint_queries": "Each checkpoint uses its own learned act_query/flow_action_query.",
        },
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 3) Build official LaWAM eval collator and Base model.
    # ------------------------------------------------------------------
    collator = build_eval_collator(cfg)
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
        f"[model] VLM on {device}; H_act shape per sample = "
        f"[{num_q}, {backend.act_query.shape[-1]}]"
    )

    base_reference: dict[tuple[Any, ...], dict[str, float]] = {}
    anchor_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 4) Base -> CL4. Same cached observation anchors; only language condition
    #    changes within each batch, and checkpoint VLM/query weights change by stage.
    # ------------------------------------------------------------------
    for stage_idx, stage in enumerate(STAGES):
        if stage_idx > 0:
            print(f"\n[{stage}] loading VLM + learned queries")
            load_vlm_queries(model, chain[stage])
            backend.vlm.eval()
        else:
            print("\n[Base] instruction-contrast H_act")

        for base_task_id in args.base_task_ids:
            ordinal = 0
            for raw_path in task_batches[int(base_task_id)]:
                raw_batch = load_raw_cache(raw_path)
                variant_batch, variant_meta = build_instruction_variant_batch(
                    collator,
                    raw_batch,
                    base_task_id=int(base_task_id),
                    negative_instruction_map=negative_instruction_map,
                )
                hact = extract_hact_only(backend, variant_batch, device)
                rows = compute_contrast_rows_for_batch(
                    suite=args.suite,
                    stage=stage,
                    hact=hact,
                    variant_meta=variant_meta,
                    negative_task_ids=args.negative_task_ids,
                    base_reference=base_reference,
                    anchor_ordinal_offset=ordinal,
                )
                anchor_rows.extend(rows)
                ordinal += len(raw_batch["samples"])

                del raw_batch, variant_batch, variant_meta, hact, rows
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            print(
                f"[{stage}] base_task={base_task_id}: "
                f"trajectories={task_meta[int(base_task_id)]['num_trajectories']} "
                f"anchors={ordinal} negatives={len(args.negative_task_ids)}"
            )

    # ------------------------------------------------------------------
    # 5) Save anchor-level results and hierarchical summaries.
    # ------------------------------------------------------------------
    anchor_fields = [
        "suite",
        "base_task_id",
        "stage",
        "anchor_ordinal",
        "trajectory_id",
        "trajectory_length",
        "trajectory_step",
        "phase",
        "phase_fraction",
        "positive_instruction",
        "negative_task_id",
        "negative_instruction",
        "negative_status",
        "hact_rel_l2",
        "hact_cosine",
        "semantic_gap",
        "l2_retention",
        "semantic_retention",
    ]
    write_csv(
        args.output_dir / "instruction_contrast_anchor_metrics.csv",
        anchor_rows,
        anchor_fields,
    )

    trajectory_rows, task_negative_rows, stage_negative_rows = summarize_hierarchical(
        anchor_rows
    )
    task_all_rows = summarize_task_all_negative(task_negative_rows)
    stage_rows = summarize_stage(task_negative_rows)

    trajectory_fields = [
        "stage",
        "base_task_id",
        "trajectory_id",
        "trajectory_length",
        "positive_instruction",
        "negative_task_id",
        "negative_instruction",
        "negative_status",
        "n_anchors",
    ] + [f"{m}_mean" for m in SUMMARY_METRICS]
    write_csv(
        args.output_dir / "instruction_contrast_trajectory_summary.csv",
        trajectory_rows,
        trajectory_fields,
    )

    task_negative_fields = [
        "stage",
        "base_task_id",
        "positive_instruction",
        "negative_task_id",
        "negative_instruction",
        "negative_status",
        "n_trajectories",
        "n_anchors",
    ]
    for metric in SUMMARY_METRICS:
        task_negative_fields += [f"{metric}_mean", f"{metric}_trajectory_std"]
    write_csv(
        args.output_dir / "instruction_contrast_task_negative_summary.csv",
        task_negative_rows,
        task_negative_fields,
    )

    stage_negative_fields = [
        "stage",
        "negative_task_id",
        "negative_instruction",
        "negative_status",
        "n_base_tasks",
    ]
    for metric in SUMMARY_METRICS:
        stage_negative_fields += [f"{metric}_macro_mean", f"{metric}_across_task_std"]
    write_csv(
        args.output_dir / "instruction_contrast_stage_negative_summary.csv",
        stage_negative_rows,
        stage_negative_fields,
    )

    task_all_fields = [
        "stage",
        "base_task_id",
        "n_negative_instructions",
    ] + [f"all_negative_{m}_mean" for m in SUMMARY_METRICS]
    write_csv(
        args.output_dir / "instruction_contrast_task_all_negative_summary.csv",
        task_all_rows,
        task_all_fields,
    )

    stage_fields = ["stage", "current_negative_task_id"]
    for category in ("all", "current", "seen_prior", "unseen"):
        stage_fields.append(f"{category}_n_task_negative_pairs")
        for metric in SUMMARY_METRICS:
            stage_fields.append(f"{category}_{metric}")
    write_csv(
        args.output_dir / "instruction_contrast_stage_summary.csv",
        stage_rows,
        stage_fields,
    )

    # Main matrices: all mismatched instructions and the stage-current new instruction.
    for metric in SUMMARY_METRICS:
        write_task_matrix(
            args.output_dir / f"matrix_all_negative_{metric}.csv",
            task_all_rows,
            args.base_task_ids,
            f"all_negative_{metric}_mean",
        )
        write_current_matrix(
            args.output_dir / f"matrix_current_negative_{metric}.csv",
            task_negative_rows,
            args.base_task_ids,
            metric,
        )

    print("\n" + "=" * 88)
    print("H_act instruction-contrast analysis complete")
    print("=" * 88)
    print("instruction map :", args.output_dir / "instruction_map.csv")
    print("coverage        :", args.output_dir / "trajectory_coverage.csv")
    print("anchor metrics  :", args.output_dir / "instruction_contrast_anchor_metrics.csv")
    print("trajectory      :", args.output_dir / "instruction_contrast_trajectory_summary.csv")
    print("task x negative :", args.output_dir / "instruction_contrast_task_negative_summary.csv")
    print("stage x negative:", args.output_dir / "instruction_contrast_stage_negative_summary.csv")
    print("stage summary   :", args.output_dir / "instruction_contrast_stage_summary.csv")
    print("main matrices   : matrix_all_negative_* / matrix_current_negative_*")
    print("=" * 88)


if __name__ == "__main__":
    main()