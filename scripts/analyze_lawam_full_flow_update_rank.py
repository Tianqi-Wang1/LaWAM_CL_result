#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import torch

FLOW_PREFIX = "policy_backend.flow."

TARGET_PATTERNS = {
    "attn_q": re.compile(r"^DiT\.transformer_blocks\.(\d+)\.attn1\.to_q\.weight$"),
    "attn_k": re.compile(r"^DiT\.transformer_blocks\.(\d+)\.attn1\.to_k\.weight$"),
    "attn_v": re.compile(r"^DiT\.transformer_blocks\.(\d+)\.attn1\.to_v\.weight$"),
    "attn_o": re.compile(r"^DiT\.transformer_blocks\.(\d+)\.attn1\.to_out\.0\.weight$"),
    "ff_in": re.compile(r"^DiT\.transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.weight$"),
    "ff_out": re.compile(r"^DiT\.transformer_blocks\.(\d+)\.ff\.net\.2\.weight$"),
}


def load_state_dict(path: Path):
    kwargs = {"map_location": "cpu"}
    try:
        obj = torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            obj = torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            obj = torch.load(path, **kwargs)

    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)
            if isinstance(nested, dict) and nested and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")

    if obj and all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[7:]: v for k, v in obj.items()}

    return obj


def canonical_flow(state: dict) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value) and str(key).startswith(FLOW_PREFIX):
            out[str(key)[len(FLOW_PREFIX):]] = value
    if not out:
        raise RuntimeError("No policy_backend.flow.* tensors found.")
    return out


def _extract_block(key: str):
    match = re.match(r"^DiT\.transformer_blocks\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def classify_flow_key(key: str) -> Tuple[str, int | None, bool]:
    for category, pattern in TARGET_PATTERNS.items():
        match = pattern.match(key)
        if match:
            return category, int(match.group(1)), True

    if re.match(r"^DiT\.transformer_blocks\.\d+\.norm", key):
        return "transformer_norm", _extract_block(key), False
    if key.startswith("DiT.timestep_encoder."):
        return "timestep_encoder", None, False
    if key.startswith("DiT.proj_out_"):
        return "output_projection", None, False
    if key.startswith("enc_vlm."):
        return "enc_vlm", None, False
    if key.startswith("enc_wm."):
        return "enc_wm", None, False
    if key.startswith("action_encoder."):
        return "action_encoder", None, False
    if key.startswith("action_decoder."):
        return "action_decoder", None, False
    if key.startswith("DiT.transformer_blocks."):
        return "transformer_other", _extract_block(key), False
    if key.startswith("DiT."):
        return "dit_other", None, False
    return "flow_other", None, False


def safe_float_tensor(x: torch.Tensor, device: torch.device):
    return x.detach().to(device=device, dtype=torch.float32, non_blocking=False)


@torch.no_grad()
def tensor_energy_stats(base: torch.Tensor, task: torch.Tensor, device: torch.device):
    if tuple(base.shape) != tuple(task.shape):
        raise RuntimeError(f"Shape mismatch: Base={tuple(base.shape)}, task={tuple(task.shape)}")

    a = safe_float_tensor(base, device)
    b = safe_float_tensor(task, device)
    delta = b - a

    base_energy = float(torch.sum(a * a).item())
    update_energy = float(torch.sum(delta * delta).item())
    base_norm = math.sqrt(max(base_energy, 0.0))
    update_norm = math.sqrt(max(update_energy, 0.0))
    rel = update_norm / base_norm if base_norm > 0 else float("nan")
    max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.abs().mean().item()) if delta.numel() else 0.0
    changed = bool(update_energy > 0.0)

    del a, b, delta
    return {
        "base_energy": base_energy,
        "update_energy": update_energy,
        "base_norm": base_norm,
        "update_norm": update_norm,
        "relative_fro": rel,
        "max_abs_update": max_abs,
        "mean_abs_update": mean_abs,
        "changed": changed,
    }


@torch.no_grad()
def exact_singular_values(base: torch.Tensor, task: torch.Tensor, device: torch.device):
    if base.ndim != 2 or task.ndim != 2:
        raise ValueError("SVD requested for a non-2D tensor.")

    a = safe_float_tensor(base, device)
    b = safe_float_tensor(task, device)
    delta = b - a
    total_energy = float(torch.sum(delta * delta).item())

    if total_energy == 0.0:
        singular = torch.zeros(min(delta.shape), dtype=torch.float32)
    else:
        singular = torch.linalg.svdvals(delta).detach().cpu()

    del a, b, delta
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return singular, total_energy


def energy_capture(singular: torch.Tensor, total_energy: float, rank: int):
    if total_energy <= 0:
        return 1.0, 0.0
    r = min(rank, singular.numel())
    captured = float(torch.sum(singular[:r] ** 2).item())
    return captured / total_energy, captured


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="LaWAM Full-Flow update localization + exact SVD intrinsic-rank analysis."
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument(
        "--task", action="append", nargs=3, metavar=("LABEL", "TASK_ID", "CHECKPOINT"), required=True
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ranks = sorted(set(args.ranks))
    if any(r <= 0 for r in ranks):
        raise ValueError("All ranks must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("LaWAM Full-Flow Update Localization + Exact SVD Analysis")
    print("=" * 100)
    print(f"Base   : {args.base}")
    print(f"Device : {device}")
    print(f"Ranks  : {ranks}")
    print(f"Output : {args.output_dir}")
    print()

    base_state = load_state_dict(args.base)
    base_flow = canonical_flow(base_state)
    base_keys = set(base_flow)
    flow_params = sum(v.numel() for v in base_flow.values())
    print(f"[Base] canonical Flow tensors={len(base_flow)}, params={flow_params:,}")

    all_per_tensor = []
    all_component = []
    all_svd_tensor = []
    all_svd_group = []
    task_overview = []

    for label, task_id_s, checkpoint_s in args.task:
        task_id = int(task_id_s)
        checkpoint = Path(checkpoint_s)

        print()
        print("=" * 100)
        print(f"[Task] {label}/T{task_id}")
        print(f"       {checkpoint}")
        print("=" * 100)

        task_state = load_state_dict(checkpoint)
        task_flow = canonical_flow(task_state)

        if set(task_flow) != base_keys:
            missing = sorted(base_keys - set(task_flow))
            extra = sorted(set(task_flow) - base_keys)
            raise RuntimeError(
                f"{label}: Flow key mismatch: missing={missing[:20]}, extra={extra[:20]}"
            )

        component_acc = defaultdict(
            lambda: {
                "num_tensors": 0,
                "num_params": 0,
                "changed_tensors": 0,
                "base_energy": 0.0,
                "update_energy": 0.0,
            }
        )

        flow_base_energy = 0.0
        flow_update_energy = 0.0
        target_update_energy = 0.0
        changed_tensors = 0

        print("[A] Computing parameter-update localization ...")
        for idx, key in enumerate(sorted(base_keys), start=1):
            base_tensor = base_flow[key]
            task_tensor = task_flow[key]
            category, block, is_target = classify_flow_key(key)
            stats = tensor_energy_stats(base_tensor, task_tensor, device)

            all_per_tensor.append(
                {
                    "task": label,
                    "task_id": task_id,
                    "key": key,
                    "component": category,
                    "block": "" if block is None else block,
                    "is_current_lora_target": int(is_target),
                    "shape": "x".join(map(str, base_tensor.shape)),
                    "num_params": base_tensor.numel(),
                    **stats,
                }
            )

            acc = component_acc[category]
            acc["num_tensors"] += 1
            acc["num_params"] += base_tensor.numel()
            acc["changed_tensors"] += int(stats["changed"])
            acc["base_energy"] += stats["base_energy"]
            acc["update_energy"] += stats["update_energy"]

            flow_base_energy += stats["base_energy"]
            flow_update_energy += stats["update_energy"]
            changed_tensors += int(stats["changed"])
            if is_target:
                target_update_energy += stats["update_energy"]

            if idx % 50 == 0 or idx == len(base_keys):
                print(f"    localization {idx}/{len(base_keys)}")

        for category, acc in sorted(component_acc.items()):
            relative = (
                math.sqrt(acc["update_energy"] / acc["base_energy"])
                if acc["base_energy"] > 0 else float("nan")
            )
            share = acc["update_energy"] / flow_update_energy if flow_update_energy > 0 else 0.0
            all_component.append(
                {
                    "task": label,
                    "task_id": task_id,
                    "component": category,
                    "num_tensors": acc["num_tensors"],
                    "changed_tensors": acc["changed_tensors"],
                    "num_params": acc["num_params"],
                    "base_energy": acc["base_energy"],
                    "update_energy": acc["update_energy"],
                    "update_fro": math.sqrt(max(acc["update_energy"], 0.0)),
                    "relative_fro": relative,
                    "share_of_total_flow_update_energy": share,
                }
            )

        target_coverage = target_update_energy / flow_update_energy if flow_update_energy > 0 else 0.0

        target_keys = [key for key in sorted(base_keys) if classify_flow_key(key)[2]]
        if len(target_keys) != 96:
            raise RuntimeError(
                f"{label}: expected exactly 96 current LoRA target weights, found {len(target_keys)}"
            )

        print(f"[B] Exact SVD on {len(target_keys)} current LoRA target matrices ...")
        svd_group_acc = defaultdict(
            lambda: {
                "total_energy": 0.0,
                "captured": {r: 0.0 for r in ranks},
                "num_matrices": 0,
            }
        )

        for idx, key in enumerate(target_keys, start=1):
            category, block, _ = classify_flow_key(key)
            singular, total_energy = exact_singular_values(base_flow[key], task_flow[key], device)

            tensor_row = {
                "task": label,
                "task_id": task_id,
                "key": key,
                "component": category,
                "block": block,
                "shape": "x".join(map(str, base_flow[key].shape)),
                "update_energy": total_energy,
                "update_fro": math.sqrt(max(total_energy, 0.0)),
                "max_possible_rank": singular.numel(),
            }

            for r in ranks:
                frac, captured = energy_capture(singular, total_energy, r)
                tensor_row[f"energy_capture_r{r}"] = frac
                for group in ("all_targets", category):
                    svd_group_acc[group]["captured"][r] += captured

            for group in ("all_targets", category):
                svd_group_acc[group]["total_energy"] += total_energy
                svd_group_acc[group]["num_matrices"] += 1

            all_svd_tensor.append(tensor_row)
            if idx % 8 == 0 or idx == len(target_keys):
                print(f"    SVD {idx}/{len(target_keys)}")

        all_target_capture = {}
        for group, acc in sorted(svd_group_acc.items()):
            row = {
                "task": label,
                "task_id": task_id,
                "group": group,
                "num_matrices": acc["num_matrices"],
                "total_update_energy": acc["total_energy"],
            }
            for r in ranks:
                capture = (
                    acc["captured"][r] / acc["total_energy"]
                    if acc["total_energy"] > 0 else 1.0
                )
                row[f"weighted_energy_capture_r{r}"] = capture
                if group == "all_targets":
                    all_target_capture[r] = capture
            all_svd_group.append(row)

        overview = {
            "task": label,
            "task_id": task_id,
            "flow_tensors": len(base_flow),
            "flow_changed_tensors": changed_tensors,
            "flow_params": flow_params,
            "flow_base_energy": flow_base_energy,
            "flow_update_energy": flow_update_energy,
            "global_relative_flow_update": (
                math.sqrt(flow_update_energy / flow_base_energy)
                if flow_base_energy > 0 else float("nan")
            ),
            "current_lora_target_update_energy": target_update_energy,
            "current_lora_target_coverage_of_total_flow_update": target_coverage,
        }

        for r in ranks:
            overview[f"target_weighted_energy_capture_r{r}"] = all_target_capture[r]
            overview[f"total_flow_representable_fraction_r{r}"] = target_coverage * all_target_capture[r]

        task_overview.append(overview)

        print()
        print(f"[{label}] summary")
        print(f"  changed Flow tensors              : {changed_tensors}/{len(base_flow)}")
        print(
            f"  current 96-target update coverage : {100 * target_coverage:.2f}% "
            f"of total Flow update energy"
        )
        for r in ranks:
            print(
                f"  rank-{r:<2d} capture within targets      : "
                f"{100 * all_target_capture[r]:6.2f}% | representable total Flow update: "
                f"{100 * target_coverage * all_target_capture[r]:6.2f}%"
            )

        del task_flow, task_state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "task_overview.csv", task_overview)
    write_csv(args.output_dir / "per_tensor_update.csv", all_per_tensor)
    write_csv(args.output_dir / "component_update_summary.csv", all_component)
    write_csv(args.output_dir / "svd_per_target_tensor.csv", all_svd_tensor)
    write_csv(args.output_dir / "svd_weighted_summary.csv", all_svd_group)

    report_path = args.output_dir / "SUMMARY.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("LaWAM Full-Flow Update Localization + Intrinsic-Rank Analysis\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Base: {args.base}\n")
        f.write(f"Ranks: {ranks}\n\n")
        f.write(
            "Interpretation:\n"
            "  target coverage = fraction of FULL Flow update energy inside the\n"
            "                    96 Linear weights used by current LoRA.\n"
            "  target rank-r capture = best rank-r SVD energy retained inside\n"
            "                          those target matrices.\n"
            "  total representable fraction = target coverage * rank-r capture.\n\n"
        )
        for row in task_overview:
            f.write(f"{row['task']}/T{row['task_id']}\n")
            f.write(
                f"  changed Flow tensors: {row['flow_changed_tensors']}/{row['flow_tensors']}\n"
            )
            f.write(
                f"  target coverage: "
                f"{100 * row['current_lora_target_coverage_of_total_flow_update']:.2f}%\n"
            )
            for r in ranks:
                f.write(
                    f"  r={r:<2d}: target capture="
                    f"{100 * row[f'target_weighted_energy_capture_r{r}']:.2f}%"
                    f", total representable="
                    f"{100 * row[f'total_flow_representable_fraction_r{r}']:.2f}%\n"
                )
            f.write("\n")

    print()
    print("=" * 100)
    print("[DONE] Saved analysis outputs:")
    for name in (
        "task_overview.csv",
        "per_tensor_update.csv",
        "component_update_summary.csv",
        "svd_per_target_tensor.csv",
        "svd_weighted_summary.csv",
        "SUMMARY.txt",
    ):
        print(f"  {args.output_dir / name}")
    print("=" * 100)


if __name__ == "__main__":
    main()