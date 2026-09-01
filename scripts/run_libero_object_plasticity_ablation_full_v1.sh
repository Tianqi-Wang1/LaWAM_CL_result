#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# LaWAM LIBERO-Object Rehearsal-Free Plasticity Ablation
#
# Reuse the SAME formal SeqFT Base checkpoint (Object tasks 0-5).
#
# Variant A: freeze_backbone
#   - freeze Qwen vision backbone
#   - freeze Qwen LLM backbone
#   - freeze embeddings
#   - KEEP Qwen vision merger trainable
#   - WAM-specific queries / QFormer / LaWM decoder / Flow remain trainable
#
# Variant B: slow_vlm_001
#   - use the ORIGINAL SeqFT freeze policy
#   - only reduce policy_backend.vlm LR from 1e-4 to 1e-6 (= 0.01x)
#   - action/world/base LR policies otherwise unchanged
#
# Default execution order:
#   freeze_backbone: CL1 -> CL4 -> full evaluation
#   slow_vlm_001 : CL1 -> CL4 -> full evaluation
#
# Per-variant resume controls:
#
#   FREEZE_BACKBONE_MODE=cl|eval|skip
#   FREEZE_BACKBONE_START_STAGE=1|2|3|4
#
#   SLOW_VLM_MODE=cl|eval|skip
#   SLOW_VLM_START_STAGE=1|2|3|4
#
# Examples:
#
#   # Default: run both, in order
#   bash scripts/run_libero_object_plasticity_ablation_full_v1.sh
#
#   # Freeze-backbone already completed; run only slow-VLM
#   FREEZE_BACKBONE_MODE=skip \
#   SLOW_VLM_MODE=cl \
#   bash scripts/run_libero_object_plasticity_ablation_full_v1.sh
#
#   # Re-evaluate both existing chains only
#   FREEZE_BACKBONE_MODE=eval \
#   SLOW_VLM_MODE=eval \
#   bash scripts/run_libero_object_plasticity_ablation_full_v1.sh
#
#   # Resume slow-VLM from CL3 only
#   FREEZE_BACKBONE_MODE=skip \
#   SLOW_VLM_MODE=cl \
#   SLOW_VLM_START_STAGE=3 \
#   bash scripts/run_libero_object_plasticity_ablation_full_v1.sh
# =============================================================================


# =============================================================================
# 0. Environment
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh

conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"


# =============================================================================
# 1. Fixed Base checkpoint and common paths
# =============================================================================

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"

FREEZE_BACKBONE_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/freeze_backbone"
SLOW_VLM_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/slow_vlm_001"

FREEZE_BACKBONE_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_object/freeze_backbone_full"
SLOW_VLM_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_object/slow_vlm_001_full"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"

mkdir -p "${FREEZE_BACKBONE_RUN_ROOT}"
mkdir -p "${SLOW_VLM_RUN_ROOT}"
mkdir -p "${FREEZE_BACKBONE_OUTPUT_ROOT}"
mkdir -p "${SLOW_VLM_OUTPUT_ROOT}"

if [ ! -f "${SUMMARY_SCRIPT}" ]; then
    echo "[ERROR] Missing summary script:"
    echo "        ${SUMMARY_SCRIPT}"
    exit 1
fi


# =============================================================================
# 2. Resources
# =============================================================================

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"

POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"

EVAL_WORKERS="${EVAL_WORKERS:-16}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"

if [ "${#TRAIN_GPU_ARRAY[@]}" -ne 4 ]; then
    echo "[ERROR] Exactly four training GPUs are required."
    echo "        TRAIN_GPUS=${TRAIN_GPUS}"
    exit 1
fi


# =============================================================================
# 3. Formal CL hyperparameters
#    Identical to the existing Object SeqFT CL1->CL4 experiment.
# =============================================================================

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"

NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"

LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

# Original LIBERO VLM LR = 1e-4.
# slow_vlm_001 uses exactly 0.01x initial VLM LR.
ORIGINAL_VLM_LR="0.0001"
SLOW_VLM_LR="0.000001"


# =============================================================================
# 4. Variant controls
# =============================================================================

FREEZE_BACKBONE_MODE="${FREEZE_BACKBONE_MODE:-cl}"
FREEZE_BACKBONE_START_STAGE="${FREEZE_BACKBONE_START_STAGE:-1}"

SLOW_VLM_MODE="${SLOW_VLM_MODE:-cl}"
SLOW_VLM_START_STAGE="${SLOW_VLM_START_STAGE:-1}"

validate_mode() {
    local name="$1"
    local mode="$2"
    local stage="$3"

    case "${mode}" in
        cl|eval|skip)
            ;;
        *)
            echo "[ERROR] ${name}_MODE must be cl, eval, or skip."
            exit 1
            ;;
    esac

    if ! [[ "${stage}" =~ ^[1-4]$ ]]; then
        echo "[ERROR] ${name}_START_STAGE must be 1, 2, 3, or 4."
        exit 1
    fi
}

validate_mode "FREEZE_BACKBONE" "${FREEZE_BACKBONE_MODE}" "${FREEZE_BACKBONE_START_STAGE}"
validate_mode "SLOW_VLM" "${SLOW_VLM_MODE}" "${SLOW_VLM_START_STAGE}"


# =============================================================================
# 5. Runtime environment
# =============================================================================

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

unset NCCL_TOPO_FILE || true
unset NCCL_GRAPH_FILE || true
unset NCCL_CONF_FILE || true
unset HFAI_NCCL_OPT_LEVEL || true


# =============================================================================
# 6. Generic helpers
# =============================================================================

find_run() {
    local root="$1"
    local pattern="$2"

    find "${root}" \
        -maxdepth 1 \
        -type d \
        -name "${pattern}" \
        | sort \
        | tail -n 1
}


verify_run() {
    local name="$1"
    local run="$2"

    if [ -z "${run}" ]; then
        echo "[ERROR] ${name} run was not found."
        exit 1
    fi

    if [ ! -f "${run}/final_model/pytorch_model.pt" ]; then
        echo "[ERROR] ${name} checkpoint missing:"
        echo "        ${run}/final_model/pytorch_model.pt"
        exit 1
    fi

    if [ ! -f "${run}/dataset_statistics.json" ]; then
        echo "[ERROR] ${name} statistics missing:"
        echo "        ${run}/dataset_statistics.json"
        exit 1
    fi

    echo "[OK] ${name}"
    echo "     run   : ${run}"
    echo "     ckpt  : ${run}/final_model/pytorch_model.pt"
    echo "     stats : ${run}/dataset_statistics.json"
}


verify_statistics() {
    local reference_stats="$1"
    local current_stats="$2"

    python - "${reference_stats}" "${current_stats}" <<'PY'
import json
import sys

reference_path, current_path = sys.argv[1:3]

with open(reference_path, "r", encoding="utf-8") as f:
    reference = json.load(f)

with open(current_path, "r", encoding="utf-8") as f:
    current = json.load(f)

for tag in reference:
    if tag not in current:
        raise RuntimeError(f"Missing embodiment tag in current statistics: {tag}")

    for section in ("action", "state"):
        if section not in reference[tag]:
            raise RuntimeError(f"Missing reference statistics section: {tag}/{section}")
        if section not in current[tag]:
            raise RuntimeError(f"Missing current statistics section: {tag}/{section}")
        if reference[tag][section] != current[tag][section]:
            raise RuntimeError(f"Normalization statistics changed: {tag}/{section}")

print("[OK] action/state normalization is identical to Base statistics.")
PY
}


# =============================================================================
# 7. Resolve EXISTING formal SeqFT Base
# =============================================================================

if [ -n "${BASE_RUN:-}" ]; then
    echo "[INFO] Using explicitly provided BASE_RUN:"
    echo "       ${BASE_RUN}"
else
    BASE_RUN=$(find_run \
        "${BASE_ROOT}" \
        '*+base_t0_5_10k_4gpu_bs32_ga2'
    )
fi

verify_run "Existing Object SeqFT Base" "${BASE_RUN}"

BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"


# =============================================================================
# 8. Variant config verification
# =============================================================================

verify_variant_config() {
    local variant="$1"
    local config_path="$2"

    if [ ! -f "${config_path}" ]; then
        echo "[ERROR] Missing config.yaml:"
        echo "        ${config_path}"
        exit 1
    fi

    python - "${variant}" "${config_path}" "${ORIGINAL_VLM_LR}" "${SLOW_VLM_LR}" <<'PY'
import math
import sys
from omegaconf import OmegaConf

variant, path, original_lr_s, slow_lr_s = sys.argv[1:5]
original_lr = float(original_lr_s)
slow_lr = float(slow_lr_s)

cfg = OmegaConf.load(path)
freeze = cfg.trainer.freeze
lr_cfg = cfg.trainer.learning_rate

if variant == "freeze_backbone":
    expected_freeze = {
        "freeze_vision_backbone": True,
        "freeze_llm_backbone": True,
        "freeze_last_llm_layer": True,
        "freeze_embedding": True,
        "unfreeze_vision_merger": True,
        "keep_llm_first_n_layers": 16,
        "unfreeze_llm_last_n_layers": -1,
        "unfreeze_lam_decoder": True,
    }
    expected_vlm_lr = original_lr

elif variant == "slow_vlm_001":
    expected_freeze = {
        "freeze_vision_backbone": False,
        "freeze_llm_backbone": False,
        "freeze_last_llm_layer": True,
        "freeze_embedding": True,
        "unfreeze_vision_merger": True,
        "keep_llm_first_n_layers": 16,
        "unfreeze_llm_last_n_layers": -1,
        "unfreeze_lam_decoder": True,
    }
    expected_vlm_lr = slow_lr

else:
    raise RuntimeError(f"Unknown variant: {variant}")

bad = []
for key, expected in expected_freeze.items():
    actual = freeze.get(key, None)
    if actual != expected:
        bad.append((f"freeze.{key}", actual, expected))

actual_vlm_lr = float(lr_cfg.vlm.lr)
if not math.isclose(actual_vlm_lr, expected_vlm_lr, rel_tol=0.0, abs_tol=1e-12):
    bad.append(("learning_rate.vlm.lr", actual_vlm_lr, expected_vlm_lr))

# These should remain exactly at the formal SeqFT values.
for key in ("action_model", "world_model"):
    actual = float(lr_cfg[key].lr)
    if not math.isclose(actual, original_lr, rel_tol=0.0, abs_tol=1e-12):
        bad.append((f"learning_rate.{key}.lr", actual, original_lr))

if bad:
    text = "\n".join(
        f"  {k}: actual={a!r}, expected={e!r}"
        for k, a, e in bad
    )
    raise RuntimeError(
        f"{variant} config mismatch:\n{text}"
    )

print(f"[OK] {variant} config verified.")
print(f"     VLM LR: {actual_vlm_lr:.8g}")
for key, value in expected_freeze.items():
    print(f"     {key}: {value}")
PY
}


# =============================================================================
# 9. Checkpoint-level variant verification
#
# freeze_backbone:
#   - all VLM tensors EXCEPT vision merger must remain bitwise unchanged
#   - merger tensors must exist and at least one should change
#
# slow_vlm_001:
#   - VLM key set must remain unchanged
#   - at least one VLM tensor should change
#
# Both variants:
#   - report WAM-specific interface changes
# =============================================================================

verify_checkpoint_variant() {
    local variant="$1"
    local previous_ckpt="$2"
    local current_ckpt="$3"
    local stage_name="$4"

    python - "${variant}" "${previous_ckpt}" "${current_ckpt}" "${stage_name}" <<'PY'
import gc
import sys
import torch

variant, prev_path, cur_path, stage = sys.argv[1:5]


def load_state(path):
    kwargs = dict(map_location="cpu")
    try:
        obj = torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            obj = torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            obj = torch.load(path, **kwargs)

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            nested = obj.get(key, None)
            if isinstance(nested, dict) and nested and any(
                torch.is_tensor(v) for v in nested.values()
            ):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint is not a state dict: {path}")

    return obj


prev = load_state(prev_path)
cur = load_state(cur_path)


def tensor_keys(prefix):
    p = {
        k for k, v in prev.items()
        if k.startswith(prefix) and torch.is_tensor(v)
    }
    c = {
        k for k, v in cur.items()
        if k.startswith(prefix) and torch.is_tensor(v)
    }
    return p, c


def exact_changed(keys):
    changed = []
    for key in sorted(keys):
        a = prev[key]
        b = cur[key]
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            changed.append(key)
        elif not torch.equal(a, b):
            changed.append(key)
    return changed



# Avoid Unicode invisible-character issues from copied expressions.
def safe_report(prefix):
    pkeys, ckeys = tensor_keys(prefix)
    missing = pkeys - ckeys
    extra = ckeys - pkeys
    common = pkeys & ckeys
    changed = exact_changed(common)
    print(
        f"  {prefix:<38s} "
        f"prev={len(pkeys):4d} cur={len(ckeys):4d} "
        f"changed={len(changed):4d} "
        f"missing={len(missing):3d} extra={len(extra):3d}"
    )
    return pkeys, ckeys, changed, missing, extra


print()
print(f"[variant-check] {variant} / {stage}")

vlm_prefix = "policy_backend.vlm."
pkeys, ckeys, vlm_changed, missing, extra = safe_report(vlm_prefix)

if not pkeys or not ckeys:
    raise RuntimeError("Could not find policy_backend.vlm tensors.")
if missing or extra:
    raise RuntimeError(
        f"{variant}/{stage}: VLM key set changed; "
        f"missing={list(sorted(missing))[:6]}, extra={list(sorted(extra))[:6]}"
    )

if variant == "freeze_backbone":
    merger_keys = {
        k for k in pkeys
        if (
            ".merger." in k
            or ".deepstack_merger_list." in k
            or k.endswith(".merger")
        )
    }

    if not merger_keys:
        raise RuntimeError(
            "freeze_backbone check could not locate vision merger tensors."
        )

    frozen_vlm_keys = pkeys - merger_keys
    frozen_changed = exact_changed(frozen_vlm_keys)
    merger_changed = exact_changed(merger_keys)

    print(
        f"  {'VLM frozen non-merger':<38s} "
        f"tensors={len(frozen_vlm_keys):4d} changed={len(frozen_changed):4d}"
    )
    print(
        f"  {'VLM trainable merger':<38s} "
        f"tensors={len(merger_keys):4d} changed={len(merger_changed):4d}"
    )

    if frozen_changed:
        raise RuntimeError(
            f"{variant}/{stage}: frozen non-merger VLM tensors changed. "
            f"Examples: {frozen_changed[:8]}"
        )

    if not merger_changed:
        raise RuntimeError(
            f"{variant}/{stage}: no vision-merger tensor changed; "
            "the intended plastic merger may not be training."
        )

    print(
        f"[OK] {stage}: backbones are bitwise stable and vision merger is plastic."
    )

elif variant == "slow_vlm_001":
    if not vlm_changed:
        raise RuntimeError(
            f"{variant}/{stage}: no VLM tensor changed; slow-VLM training "
            "appears inactive."
        )

    print(
        f"[OK] {stage}: VLM is plastic under the reduced LR "
        f"({len(vlm_changed)} tensors changed)."
    )

else:
    raise RuntimeError(f"Unknown variant: {variant}")

# WAM-specific modules: report only.
for prefix in (
    "policy_backend.act_query",
    "policy_backend.flow_action_query",
    "policy_backend.vlm_to_lam.",
    "policy_backend.lam.decoder.",
    "policy_backend.flow.",
):
    safe_report(prefix)

del prev, cur
gc.collect()
PY
}


# =============================================================================
# 10. Evaluation helper
# =============================================================================

run_eval_stage() {
    local master_dir="$1"
    local manifest="$2"
    local stage="$3"
    local train_run="$4"
    local task_ids="$5"
    local alias="$6"

    local ckpt="${train_run}/final_model/pytorch_model.pt"

    echo
    echo "=========================================================="
    echo " Evaluating ${stage}"
    echo "=========================================================="
    echo "Checkpoint:"
    echo "  ${ckpt}"
    echo "Task IDs:"
    echo "  ${task_ids}"
    echo "=========================================================="
    echo

    SUITES="libero_object" \
    TASK_IDS="${task_ids}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${EVAL_WORKERS}" \
    GPU_IDS="${POLICY_GPU}" \
    EVAL_GPU_IDS="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${master_dir}" \
    LIBERO_CKPT_ALIAS="${alias}" \
    bash \
    examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh \
    "${ckpt}"

    local eval_timestamp_dir
    eval_timestamp_dir=$(find \
        "${master_dir}/${alias}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        | sort \
        | tail -n 1
    )

    if [ -z "${eval_timestamp_dir}" ]; then
        echo "[ERROR] Evaluation output not found:"
        echo "        ${master_dir}/${alias}"
        exit 1
    fi

    local suite_dir="${eval_timestamp_dir}/suites/libero_object"

    if [ ! -f "${suite_dir}/episodes.jsonl" ]; then
        echo "[ERROR] Missing episodes.jsonl:"
        echo "        ${suite_dir}/episodes.jsonl"
        exit 1
    fi

    if [ ! -f "${suite_dir}/summary.json" ]; then
        echo "[ERROR] Missing summary.json:"
        echo "        ${suite_dir}/summary.json"
        exit 1
    fi

    read -r -a TASK_ARRAY <<< "${task_ids}"

    python "${SUMMARY_SCRIPT}" \
        --run-dir "${suite_dir}" \
        --task-ids "${TASK_ARRAY[@]}" \
        --expected-trials "${NUM_TRIALS}"

    printf \
        "%s\t%s\t%s\t%s\n" \
        "${stage}" \
        "${train_run}" \
        "${suite_dir}" \
        "${task_ids}" \
        >> "${manifest}"

    echo "[OK] ${stage} evaluation completed."
}


build_performance_matrix() {
    local manifest="$1"
    local master_dir="$2"

    python - "${manifest}" "${master_dir}" "${NUM_TRIALS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
master_dir = Path(sys.argv[2])
expected_trials = int(sys.argv[3])

with manifest_path.open("r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f, delimiter="\t"))

long_rows = []
stage_results = {}
stage_summaries = []

for item in manifest_rows:
    stage = item["stage"]
    run_dir = Path(item["eval_run_dir"])
    per_task_path = run_dir / "per_task_summary.json"

    with per_task_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    stage_results[stage] = {}

    for row in rows:
        task_id = int(row["task_id"])
        sr = float(row["success_rate"])
        successes = int(row["successes"])
        trials = int(row["trials"])

        if trials != expected_trials:
            raise RuntimeError(
                f"{stage}/task{task_id}: expected {expected_trials} trials, got {trials}"
            )

        stage_results[stage][task_id] = sr

        long_rows.append(
            {
                "stage": stage,
                "task_id": task_id,
                "task_description": row["task_description"],
                "successes": successes,
                "trials": trials,
                "success_rate": sr,
            }
        )

    total_success = sum(int(row["successes"]) for row in rows)
    total_trials = sum(int(row["trials"]) for row in rows)
    mean_task_sr = sum(float(row["success_rate"]) for row in rows) / len(rows)

    stage_summaries.append(
        {
            "stage": stage,
            "num_tasks": len(rows),
            "successes": total_success,
            "trials": total_trials,
            "overall_success_rate": total_success / total_trials,
            "mean_task_success_rate": mean_task_sr,
        }
    )

long_path = master_dir / "cl_per_task_long.csv"
with long_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stage",
            "task_id",
            "task_description",
            "successes",
            "trials",
            "success_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(long_rows)

stage_summary_path = master_dir / "cl_stage_summary.csv"
with stage_summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stage",
            "num_tasks",
            "successes",
            "trials",
            "overall_success_rate",
            "mean_task_success_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(stage_summaries)

stage_order = ["Base", "CL1", "CL2", "CL3", "CL4"]

matrix_path = master_dir / "cl_performance_matrix.csv"
with matrix_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = (
        ["stage"]
        + [f"task_{task_id}" for task_id in range(10)]
        + ["mean_seen_task_sr"]
    )

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for stage in stage_order:
        result = stage_results[stage]
        row = {"stage": stage}

        for task_id in range(10):
            row[f"task_{task_id}"] = (
                f"{result[task_id]:.4f}"
                if task_id in result
                else ""
            )

        row["mean_seen_task_sr"] = (
            f"{sum(result.values()) / len(result):.4f}"
        )
        writer.writerow(row)

print()
print("Stage | " + " | ".join(f"T{i}" for i in range(10)) + " | Mean")
print("-" * 104)

for stage in stage_order:
    result = stage_results[stage]
    values = [
        f"{result[i]:.2f}" if i in result else "-"
        for i in range(10)
    ]
    mean_sr = sum(result.values()) / len(result)
    print(
        f"{stage:>4s} | "
        + " | ".join(f"{value:>4s}" for value in values)
        + f" | {mean_sr:.4f}"
    )

print()
print("[OK] Saved:")
print(matrix_path)
print(long_path)
print(stage_summary_path)
PY
}


# =============================================================================
# 11. Train one variant
# =============================================================================

train_variant() {
    local variant="$1"
    local run_root="$2"
    local run_suffix="$3"
    local start_stage="$4"

    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES=4

    local prev_run

    if [ "${start_stage}" -eq 1 ]; then
        prev_run="${BASE_RUN}"
    else
        local prev_stage=$((start_stage - 1))
        local prev_task_id=$((prev_stage + 5))
        local prev_run_id="cl${prev_stage}_t${prev_task_id}_2k_4gpu_bs32_ga2_${run_suffix}"

        prev_run=$(find_run \
            "${run_root}" \
            "*+${prev_run_id}"
        )

        verify_run "${variant} existing CL${prev_stage}" "${prev_run}"
        verify_variant_config "${variant}" "${prev_run}/config.yaml"
    fi

    for stage in $(seq "${start_stage}" 4); do
        local task_id=$((stage + 5))
        local run_id="cl${stage}_t${task_id}_2k_4gpu_bs32_ga2_${run_suffix}"

        local prev_ckpt="${prev_run}/final_model/pytorch_model.pt"
        local prev_stats="${prev_run}/dataset_statistics.json"

        if [ ! -f "${prev_ckpt}" ]; then
            echo "[ERROR] Previous checkpoint missing:"
            echo "        ${prev_ckpt}"
            exit 1
        fi
        if [ ! -f "${prev_stats}" ]; then
            echo "[ERROR] Previous statistics missing:"
            echo "        ${prev_stats}"
            exit 1
        fi

        echo
        echo "=========================================================="
        echo " ${variant}: Starting CL${stage}"
        echo "=========================================================="
        echo "Task        : object ${task_id}"
        echo "Previous    : ${prev_run}"
        echo "New run ID  : ${run_id}"
        echo "=========================================================="
        echo

        common_args=(
            "--run_root_dir=${run_root}"
            "--run_id=${run_id}"

            "--datasets.vla_data.cl_suite=libero_object"
            "--datasets.vla_data.cl_task_ids=[${task_id}]"

            "--datasets.vla_data.use_task_filtered_statistics=false"
            "--trainer.use_pretrained_dataset_statistics=true"
            "--trainer.pretrained_checkpoint=${prev_ckpt}"

            "--trainer.freeze.freeze_last_llm_layer=true"
            "--trainer.freeze.freeze_embedding=true"
            "--trainer.freeze.keep_llm_first_n_layers=16"
            "--trainer.freeze.unfreeze_llm_last_n_layers=-1"
            "--trainer.freeze.unfreeze_lam_decoder=true"

            "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
            "--datasets.vla_data.num_workers=${NUM_WORKERS}"
            "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}"
            "--datasets.vla_data.persistent_workers=true"

            "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
            "--trainer.max_train_steps=${MAX_TRAIN_STEPS}"
            "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}"

            "--trainer.logging_frequency=${LOGGING_FREQUENCY}"
            "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}"
            "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}"
            "--trainer.save_interval=${SAVE_INTERVAL}"
        )

        variant_args=()

        case "${variant}" in
            freeze_backbone)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=true"
                    "--trainer.freeze.freeze_llm_backbone=true"
                    "--trainer.freeze.unfreeze_vision_merger=true"
                    "--trainer.learning_rate.vlm.lr=${ORIGINAL_VLM_LR}"
                )
                ;;
            slow_vlm_001)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=false"
                    "--trainer.freeze.freeze_llm_backbone=false"
                    "--trainer.freeze.unfreeze_vision_merger=true"
                    "--trainer.learning_rate.vlm.lr=${SLOW_VLM_LR}"
                )
                ;;
            *)
                echo "[ERROR] Unknown variant: ${variant}"
                exit 1
                ;;
        esac

        bash train_lawam.sh \
            "${common_args[@]}" \
            "${variant_args[@]}"

        local current_run
        current_run=$(find_run \
            "${run_root}" \
            "*+${run_id}"
        )

        verify_run "${variant} CL${stage}" "${current_run}"

        echo "[INFO] ${variant}: verifying normalization after CL${stage}..."
        verify_statistics \
            "${BASE_STATS}" \
            "${current_run}/dataset_statistics.json"

        echo "[INFO] ${variant}: verifying config after CL${stage}..."
        verify_variant_config \
            "${variant}" \
            "${current_run}/config.yaml"

        echo "[INFO] ${variant}: verifying checkpoint behavior after CL${stage}..."
        verify_checkpoint_variant \
            "${variant}" \
            "${prev_ckpt}" \
            "${current_run}/final_model/pytorch_model.pt" \
            "CL${stage}"

        prev_run="${current_run}"
    done

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
}


# =============================================================================
# 12. Resolve one variant's full chain
# =============================================================================

resolve_variant_chain() {
    local variant="$1"
    local run_root="$2"
    local run_suffix="$3"

    V_CL1_RUN=$(find_run "${run_root}" "*+cl1_t6_2k_4gpu_bs32_ga2_${run_suffix}")
    V_CL2_RUN=$(find_run "${run_root}" "*+cl2_t7_2k_4gpu_bs32_ga2_${run_suffix}")
    V_CL3_RUN=$(find_run "${run_root}" "*+cl3_t8_2k_4gpu_bs32_ga2_${run_suffix}")
    V_CL4_RUN=$(find_run "${run_root}" "*+cl4_t9_2k_4gpu_bs32_ga2_${run_suffix}")

    verify_run "${variant} CL1" "${V_CL1_RUN}"
    verify_run "${variant} CL2" "${V_CL2_RUN}"
    verify_run "${variant} CL3" "${V_CL3_RUN}"
    verify_run "${variant} CL4" "${V_CL4_RUN}"

    for run in "${V_CL1_RUN}" "${V_CL2_RUN}" "${V_CL3_RUN}" "${V_CL4_RUN}"; do
        verify_statistics "${BASE_STATS}" "${run}/dataset_statistics.json"
        verify_variant_config "${variant}" "${run}/config.yaml"
    done
}


# =============================================================================
# 13. Evaluate one complete variant
# =============================================================================

evaluate_variant() {
    local variant="$1"
    local output_root="$2"
    local alias_prefix="$3"

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true

    export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
    export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
    export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local master_dir="${output_root}/cl_full_${timestamp}"
    mkdir -p "${master_dir}"

    local manifest="${master_dir}/eval_manifest.tsv"
    printf "stage\tcheckpoint_run\teval_run_dir\ttask_ids\n" > "${manifest}"

    echo
    echo "=========================================================="
    echo " ${variant}: full sequential evaluation"
    echo "=========================================================="
    echo "Policy GPU    : ${POLICY_GPU}"
    echo "Evaluator GPU : ${EVAL_GPU}"
    echo "Workers       : ${EVAL_WORKERS}"
    echo "Trials / task : ${NUM_TRIALS}"
    echo "Output        : ${master_dir}"
    echo "=========================================================="
    echo

    run_eval_stage \
        "${master_dir}" "${manifest}" \
        "Base" "${BASE_RUN}" \
        "0 1 2 3 4 5" \
        "${alias_prefix}_base_t0_5_10k"

    run_eval_stage \
        "${master_dir}" "${manifest}" \
        "CL1" "${V_CL1_RUN}" \
        "0 1 2 3 4 5 6" \
        "${alias_prefix}_cl1_t6_2k"

    run_eval_stage \
        "${master_dir}" "${manifest}" \
        "CL2" "${V_CL2_RUN}" \
        "0 1 2 3 4 5 6 7" \
        "${alias_prefix}_cl2_t7_2k"

    run_eval_stage \
        "${master_dir}" "${manifest}" \
        "CL3" "${V_CL3_RUN}" \
        "0 1 2 3 4 5 6 7 8" \
        "${alias_prefix}_cl3_t8_2k"

    run_eval_stage \
        "${master_dir}" "${manifest}" \
        "CL4" "${V_CL4_RUN}" \
        "0 1 2 3 4 5 6 7 8 9" \
        "${alias_prefix}_cl4_t9_2k"

    build_performance_matrix "${manifest}" "${master_dir}"

    echo
    echo "[OK] ${variant} evaluation complete:"
    echo "     ${master_dir}"
    echo
    echo "Core result:"
    echo "     ${master_dir}/cl_performance_matrix.csv"
}


# =============================================================================
# 14. Run one variant end-to-end
# =============================================================================

run_variant() {
    local variant="$1"
    local mode="$2"
    local start_stage="$3"
    local run_root="$4"
    local run_suffix="$5"
    local output_root="$6"
    local alias_prefix="$7"

    if [ "${mode}" = "skip" ]; then
        echo
        echo "[SKIP] ${variant}"
        return
    fi

    echo
    echo
    echo "################################################################################"
    echo "# VARIANT: ${variant}"
    echo "################################################################################"
    echo

    if [ "${mode}" = "cl" ]; then
        train_variant \
            "${variant}" \
            "${run_root}" \
            "${run_suffix}" \
            "${start_stage}"
    fi

    resolve_variant_chain \
        "${variant}" \
        "${run_root}" \
        "${run_suffix}"

    # Final checkpoint-behavior check against the SAME Base.
    verify_checkpoint_variant \
        "${variant}" \
        "${BASE_CKPT}" \
        "${V_CL4_RUN}/final_model/pytorch_model.pt" \
        "CL4_vs_Base"

    evaluate_variant \
        "${variant}" \
        "${output_root}" \
        "${alias_prefix}"
}


# =============================================================================
# 15. Master log and experiment summary
# =============================================================================

LOG_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/plasticity_ablation_logs"
mkdir -p "${LOG_ROOT}"

MASTER_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/plasticity_ablation_${MASTER_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo
echo "=========================================================="
echo " LaWAM Object Plasticity Ablation"
echo "=========================================================="
echo "Existing Base:"
echo "  ${BASE_RUN}"
echo
echo "Variant A: freeze_backbone"
echo "  mode        : ${FREEZE_BACKBONE_MODE}"
echo "  start stage : ${FREEZE_BACKBONE_START_STAGE}"
echo "  checkpoints : ${FREEZE_BACKBONE_RUN_ROOT}"
echo "  definition  : freeze vision+LLM backbones; train vision merger"
echo
echo "Variant B: slow_vlm_001"
echo "  mode        : ${SLOW_VLM_MODE}"
echo "  start stage : ${SLOW_VLM_START_STAGE}"
echo "  checkpoints : ${SLOW_VLM_RUN_ROOT}"
echo "  VLM LR      : ${SLOW_VLM_LR} (0.01 x ${ORIGINAL_VLM_LR})"
echo
echo "Common CL training:"
echo "  GPUs        : ${TRAIN_GPUS}"
echo "  batch/GPU   : ${PER_DEVICE_BATCH_SIZE}"
echo "  grad accum  : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  global batch: $((PER_DEVICE_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"
echo "  steps/stage : ${MAX_TRAIN_STEPS}"
echo
echo "Evaluation:"
echo "  policy GPU  : ${POLICY_GPU}"
echo "  sim GPU     : ${EVAL_GPU}"
echo "  workers     : ${EVAL_WORKERS}"
echo "  trials/task : ${NUM_TRIALS}"
echo "=========================================================="
echo


# =============================================================================
# 16. Execute variants IN ORDER
# =============================================================================

run_variant \
    "freeze_backbone" \
    "${FREEZE_BACKBONE_MODE}" \
    "${FREEZE_BACKBONE_START_STAGE}" \
    "${FREEZE_BACKBONE_RUN_ROOT}" \
    "freeze_backbone" \
    "${FREEZE_BACKBONE_OUTPUT_ROOT}" \
    "freeze_backbone"

run_variant \
    "slow_vlm_001" \
    "${SLOW_VLM_MODE}" \
    "${SLOW_VLM_START_STAGE}" \
    "${SLOW_VLM_RUN_ROOT}" \
    "slow_vlm_001" \
    "${SLOW_VLM_OUTPUT_ROOT}" \
    "slow_vlm_001"


# =============================================================================
# 17. Complete
# =============================================================================

echo
echo "=========================================================="
echo " LaWAM Object Plasticity Ablation completed"
echo "=========================================================="
echo "freeze_backbone results:"
echo "  ${FREEZE_BACKBONE_OUTPUT_ROOT}"
echo
echo "slow_vlm_001 results:"
echo "  ${SLOW_VLM_OUTPUT_ROOT}"
echo
echo "Master log:"
echo "  ${MASTER_LOG}"
echo "=========================================================="