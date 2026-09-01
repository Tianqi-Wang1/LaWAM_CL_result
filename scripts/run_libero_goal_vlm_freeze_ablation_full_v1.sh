#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# LaWAM LIBERO-Goal VLM Freeze Ablation
#
# Reuse the SAME formal SeqFT Base checkpoint (Goal tasks 0-5).
# Existing SeqFT CL results are NOT retrained here.
#
# Variant A: strict_freeze_vlm
#   Frozen during CL:
#     - Qwen vision backbone
#     - Qwen vision merger
#     - Qwen LLM backbone
#     - Qwen embeddings
#
#   Still trainable:
#     - act_query
#     - flow_action_query
#     - VLMToLAM / QFormer
#     - LaWM decoder
#     - Flow action head
#
# Variant B: freeze_backbone
#   Frozen during CL:
#     - Qwen vision backbone
#     - Qwen LLM backbone
#     - Qwen embeddings
#
#   Trainable:
#     - Qwen vision merger
#     - act_query
#     - flow_action_query
#     - VLMToLAM / QFormer
#     - LaWM decoder
#     - Flow action head
#
# Sequential protocol:
#   Base : Goal 0-5 (reuse existing 10K SeqFT Base)
#   CL1  : Goal 6, 2K
#   CL2  : Goal 7, 2K
#   CL3  : Goal 8, 2K
#   CL4  : Goal 9, 2K
#
# The script performs, for each variant:
#   1) sequential CL1 -> CL4 training
#   2) Base-statistics consistency checks
#   3) freeze-config checks
#   4) checkpoint-level parameter invariance/plasticity checks
#   5) full Base/CL1/CL2/CL3/CL4 LIBERO-Goal evaluation
#   6) CL performance-matrix aggregation
#   7) optional FWT/NBT/AUC computation
#
# Default:
#   bash scripts/run_libero_goal_vlm_freeze_ablation_full_v1.sh
#
# Run only Strict Freeze-VLM:
#   STRICT_FREEZE_VLM_MODE=cl \
#   FREEZE_BACKBONE_MODE=skip \
#   bash scripts/run_libero_goal_vlm_freeze_ablation_full_v1.sh
#
# Run only Freeze-Backbone:
#   STRICT_FREEZE_VLM_MODE=skip \
#   FREEZE_BACKBONE_MODE=cl \
#   bash scripts/run_libero_goal_vlm_freeze_ablation_full_v1.sh
#
# Resume Strict Freeze-VLM from CL3:
#   STRICT_FREEZE_VLM_MODE=cl \
#   STRICT_FREEZE_VLM_START_STAGE=3 \
#   FREEZE_BACKBONE_MODE=skip \
#   bash scripts/run_libero_goal_vlm_freeze_ablation_full_v1.sh
#
# Evaluate existing chains only:
#   STRICT_FREEZE_VLM_MODE=eval \
#   FREEZE_BACKBONE_MODE=eval \
#   bash scripts/run_libero_goal_vlm_freeze_ablation_full_v1.sh
#
# Disable FWT/NBT/AUC computation:
#   COMPUTE_CL_METRICS=false bash ...
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
# 1. Existing Goal Base and experiment roots
# =============================================================================

# Reuse the SAME formal vanilla SeqFT Base checkpoint.
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

STRICT_FREEZE_VLM_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/freeze_vlm"
FREEZE_BACKBONE_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/freeze_backbone"

STRICT_FREEZE_VLM_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/freeze_vlm_full"
FREEZE_BACKBONE_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/freeze_backbone_full"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
CL_METRICS_SCRIPT="${ROOT}/scripts/compute_libero_cl_metrics.py"

mkdir -p "${STRICT_FREEZE_VLM_RUN_ROOT}"
mkdir -p "${FREEZE_BACKBONE_RUN_ROOT}"
mkdir -p "${STRICT_FREEZE_VLM_OUTPUT_ROOT}"
mkdir -p "${FREEZE_BACKBONE_OUTPUT_ROOT}"

if [ ! -f "${SUMMARY_SCRIPT}" ]; then
    echo "[ERROR] Missing summary script:"
    echo "        ${SUMMARY_SCRIPT}"
    exit 1
fi


# =============================================================================
# 2. Resource configuration
# =============================================================================

# Training: exactly four physical GPUs.
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"

# Evaluation: one policy GPU + one simulator/evaluator GPU.
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
#    Keep identical to the existing Goal SeqFT CL protocol.
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
# Do not save periodic checkpoints during each 2K CL stage.
# The trainer still writes <run_dir>/final_model/pytorch_model.pt at the end.
# Setting the periodic interval beyond max_train_steps prevents step checkpoints.
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"

# Formal SeqFT VLM/action/world LR.
ORIGINAL_LR="${ORIGINAL_LR:-0.0001}"


# =============================================================================
# 4. Variant controls
# =============================================================================

STRICT_FREEZE_VLM_MODE="${STRICT_FREEZE_VLM_MODE:-cl}"
STRICT_FREEZE_VLM_START_STAGE="${STRICT_FREEZE_VLM_START_STAGE:-1}"

FREEZE_BACKBONE_MODE="${FREEZE_BACKBONE_MODE:-cl}"
FREEZE_BACKBONE_START_STAGE="${FREEZE_BACKBONE_START_STAGE:-1}"

COMPUTE_CL_METRICS="${COMPUTE_CL_METRICS:-true}"


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


validate_bool() {
    local name="$1"
    local value="$2"

    case "${value}" in
        true|false)
            ;;
        *)
            echo "[ERROR] ${name} must be true or false."
            exit 1
            ;;
    esac
}


validate_mode \
    "STRICT_FREEZE_VLM" \
    "${STRICT_FREEZE_VLM_MODE}" \
    "${STRICT_FREEZE_VLM_START_STAGE}"

validate_mode \
    "FREEZE_BACKBONE" \
    "${FREEZE_BACKBONE_MODE}" \
    "${FREEZE_BACKBONE_START_STAGE}"

validate_bool \
    "COMPUTE_CL_METRICS" \
    "${COMPUTE_CL_METRICS}"

if [ "${COMPUTE_CL_METRICS}" = "true" ] && [ ! -f "${CL_METRICS_SCRIPT}" ]; then
    echo "[ERROR] COMPUTE_CL_METRICS=true but script is missing:"
    echo "        ${CL_METRICS_SCRIPT}"
    echo
    echo "Place compute_libero_cl_metrics.py under ${ROOT}/scripts/"
    echo "or launch with COMPUTE_CL_METRICS=false."
    exit 1
fi


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
# 7. Locate EXISTING formal Goal SeqFT Base
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

verify_run "Existing Goal SeqFT Base" "${BASE_RUN}"

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

    python - "${variant}" "${config_path}" "${ORIGINAL_LR}" <<'PY'
import math
import sys
from omegaconf import OmegaConf

variant, path, original_lr_s = sys.argv[1:4]
original_lr = float(original_lr_s)

cfg = OmegaConf.load(path)
freeze = cfg.trainer.freeze
lr_cfg = cfg.trainer.learning_rate

if variant == "strict_freeze_vlm":
    expected_freeze = {
        "freeze_vision_backbone": True,
        "freeze_llm_backbone": True,
        "freeze_last_llm_layer": True,
        "freeze_embedding": True,
        "unfreeze_vision_merger": False,
        "keep_llm_first_n_layers": 16,
        "unfreeze_llm_last_n_layers": -1,
        "unfreeze_lam_decoder": True,
    }

elif variant == "freeze_backbone":
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

else:
    raise RuntimeError(f"Unknown variant: {variant}")

bad = []

for key, expected in expected_freeze.items():
    actual = freeze.get(key, None)
    if actual != expected:
        bad.append((f"freeze.{key}", actual, expected))

# For both variants we retain the original LR settings.
for key in ("vlm", "action_model", "world_model"):
    actual = float(lr_cfg[key].lr)
    if not math.isclose(actual, original_lr, rel_tol=0.0, abs_tol=1e-12):
        bad.append((f"learning_rate.{key}.lr", actual, original_lr))

if bad:
    text = "\n".join(
        f"  {k}: actual={a!r}, expected={e!r}"
        for k, a, e in bad
    )
    raise RuntimeError(f"{variant} config mismatch:\n{text}")

print(f"[OK] {variant} config verified.")
for key, value in expected_freeze.items():
    print(f"     {key}: {value}")

for key in ("vlm", "action_model", "world_model"):
    print(f"     learning_rate.{key}.lr: {float(lr_cfg[key].lr):.8g}")
PY
}


# =============================================================================
# 9. Checkpoint-level variant verification
#
# strict_freeze_vlm:
#   HARD ASSERTION:
#     all policy_backend.vlm.* tensors remain bitwise unchanged.
#
# freeze_backbone:
#   HARD ASSERTION:
#     all VLM tensors EXCEPT vision merger remain bitwise unchanged.
#     at least one vision-merger tensor changes.
#
# Both:
#   WAM-specific modules are report-only.
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
        obj = torch.load(
            path,
            weights_only=True,
            mmap=True,
            **kwargs,
        )
    except TypeError:
        try:
            obj = torch.load(
                path,
                weights_only=True,
                **kwargs,
            )
        except TypeError:
            obj = torch.load(path, **kwargs)

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            nested = obj.get(key, None)
            if (
                isinstance(nested, dict)
                and nested
                and any(torch.is_tensor(v) for v in nested.values())
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
        f"missing={list(sorted(missing))[:6]}, "
        f"extra={list(sorted(extra))[:6]}"
    )


if variant == "strict_freeze_vlm":
    if vlm_changed:
        raise RuntimeError(
            f"{variant}/{stage}: frozen VLM tensors changed. "
            f"Examples: {vlm_changed[:8]}"
        )

    print(
        f"[OK] {stage}: entire policy_backend.vlm is bitwise unchanged."
    )


elif variant == "freeze_backbone":
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
        f"tensors={len(frozen_vlm_keys):4d} "
        f"changed={len(frozen_changed):4d}"
    )

    print(
        f"  {'VLM trainable merger':<38s} "
        f"tensors={len(merger_keys):4d} "
        f"changed={len(merger_changed):4d}"
    )

    if frozen_changed:
        raise RuntimeError(
            f"{variant}/{stage}: frozen non-merger VLM tensors changed. "
            f"Examples: {frozen_changed[:8]}"
        )

    if not merger_changed:
        raise RuntimeError(
            f"{variant}/{stage}: no vision-merger tensor changed; "
            "the intended trainable merger may not be training."
        )

    print(
        f"[OK] {stage}: VLM backbones are bitwise stable and merger is plastic."
    )

else:
    raise RuntimeError(f"Unknown variant: {variant}")


# Report WAM-specific modules only.
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
# 10. Training one variant
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
        echo " ${variant}: Starting Goal CL${stage}"
        echo "=========================================================="
        echo "Task        : goal ${task_id}"
        echo "Previous    : ${prev_run}"
        echo "Previous ckpt:"
        echo "  ${prev_ckpt}"
        echo "New run ID  : ${run_id}"
        echo "=========================================================="
        echo

        common_args=(
            "--run_root_dir=${run_root}"
            "--run_id=${run_id}"

            "--datasets.vla_data.cl_suite=libero_goal"
            "--datasets.vla_data.cl_task_ids=[${task_id}]"

            "--datasets.vla_data.use_task_filtered_statistics=false"
            "--trainer.use_pretrained_dataset_statistics=true"
            "--trainer.pretrained_checkpoint=${prev_ckpt}"

            "--trainer.freeze.freeze_last_llm_layer=true"
            "--trainer.freeze.freeze_embedding=true"
            "--trainer.freeze.keep_llm_first_n_layers=16"
            "--trainer.freeze.unfreeze_llm_last_n_layers=-1"
            "--trainer.freeze.unfreeze_lam_decoder=true"

            "--trainer.learning_rate.vlm.lr=${ORIGINAL_LR}"
            "--trainer.learning_rate.action_model.lr=${ORIGINAL_LR}"
            "--trainer.learning_rate.world_model.lr=${ORIGINAL_LR}"

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

            strict_freeze_vlm)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=true"
                    "--trainer.freeze.freeze_llm_backbone=true"
                    "--trainer.freeze.unfreeze_vision_merger=false"
                )
                ;;

            freeze_backbone)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=true"
                    "--trainer.freeze.freeze_llm_backbone=true"
                    "--trainer.freeze.unfreeze_vision_merger=true"
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

        echo
        echo "[INFO] ${variant}: verifying Base normalization after CL${stage}..."
        verify_statistics \
            "${BASE_STATS}" \
            "${current_run}/dataset_statistics.json"

        echo
        echo "[INFO] ${variant}: verifying saved config after CL${stage}..."
        verify_variant_config \
            "${variant}" \
            "${current_run}/config.yaml"

        echo
        echo "[INFO] ${variant}: verifying checkpoint behavior after CL${stage}..."
        verify_checkpoint_variant \
            "${variant}" \
            "${prev_ckpt}" \
            "${current_run}/final_model/pytorch_model.pt" \
            "CL${stage}"

        echo
        echo "[OK] ${variant} Goal CL${stage} completed and verified."
        echo "     ${current_run}"
        echo

        prev_run="${current_run}"
    done

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
}


# =============================================================================
# 11. Resolve a complete variant chain
# =============================================================================

resolve_variant_chain() {
    local variant="$1"
    local run_root="$2"
    local run_suffix="$3"

    V_CL1_RUN=$(find_run \
        "${run_root}" \
        "*+cl1_t6_2k_4gpu_bs32_ga2_${run_suffix}"
    )

    V_CL2_RUN=$(find_run \
        "${run_root}" \
        "*+cl2_t7_2k_4gpu_bs32_ga2_${run_suffix}"
    )

    V_CL3_RUN=$(find_run \
        "${run_root}" \
        "*+cl3_t8_2k_4gpu_bs32_ga2_${run_suffix}"
    )

    V_CL4_RUN=$(find_run \
        "${run_root}" \
        "*+cl4_t9_2k_4gpu_bs32_ga2_${run_suffix}"
    )

    verify_run "${variant} CL1" "${V_CL1_RUN}"
    verify_run "${variant} CL2" "${V_CL2_RUN}"
    verify_run "${variant} CL3" "${V_CL3_RUN}"
    verify_run "${variant} CL4" "${V_CL4_RUN}"

    for run in \
        "${V_CL1_RUN}" \
        "${V_CL2_RUN}" \
        "${V_CL3_RUN}" \
        "${V_CL4_RUN}"
    do
        verify_statistics \
            "${BASE_STATS}" \
            "${run}/dataset_statistics.json"

        verify_variant_config \
            "${variant}" \
            "${run}/config.yaml"
    done
}


# =============================================================================
# 12. Evaluation helper
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

    SUITES="libero_goal" \
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

    local suite_dir="${eval_timestamp_dir}/suites/libero_goal"

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

    echo
    echo "[INFO] Aggregating ${stage} per-task results..."

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

    echo
    echo "[OK] ${stage} evaluation completed."
    echo "     ${suite_dir}"
}


# =============================================================================
# 13. Build SR matrix
# =============================================================================

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
                f"{stage}/task{task_id}: "
                f"expected {expected_trials} trials, got {trials}"
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
    mean_task_sr = (
        sum(float(row["success_rate"]) for row in rows)
        / len(rows)
    )

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

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

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
print(
    "Stage | "
    + " | ".join(f"T{i}" for i in range(10))
    + " | Mean"
)
print("-" * 104)

for stage in stage_order:
    result = stage_results[stage]

    values = [
        f"{result[i]:.2f}"
        if i in result
        else "-"
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
# 14. Optional FWT / NBT / AUC computation
# =============================================================================

compute_cl_metrics() {
    local variant="$1"
    local master_dir="$2"

    if [ "${COMPUTE_CL_METRICS}" != "true" ]; then
        echo "[INFO] COMPUTE_CL_METRICS=false; skipping FWT/NBT/AUC."
        return
    fi

    local matrix_path="${master_dir}/cl_performance_matrix.csv"
    local metrics_dir="${master_dir}/cl_metrics"

    echo
    echo "=========================================================="
    echo " Computing ${variant} FWT / NBT / AUC"
    echo "=========================================================="
    echo "Matrix:"
    echo "  ${matrix_path}"
    echo "=========================================================="
    echo

    python "${CL_METRICS_SCRIPT}" \
        "${matrix_path}" \
        --names "${variant}" \
        --base-tasks 0 1 2 3 4 5 \
        --cl-tasks 6 7 8 9 \
        --output-dir "${metrics_dir}"

    echo
    echo "[OK] CL metrics:"
    echo "     ${metrics_dir}/cl_metrics_summary.csv"
}


# =============================================================================
# 15. Evaluate a complete variant
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

    printf \
        "stage\tcheckpoint_run\teval_run_dir\ttask_ids\n" \
        > "${manifest}"

    echo
    echo "=========================================================="
    echo " ${variant}: full LIBERO-Goal sequential evaluation"
    echo "=========================================================="
    echo "Policy GPU    : ${POLICY_GPU}"
    echo "Evaluator GPU : ${EVAL_GPU}"
    echo "Workers       : ${EVAL_WORKERS}"
    echo "Trials / task : ${NUM_TRIALS}"
    echo "Save videos   : ${SAVE_VIDEOS}"
    echo "Output        : ${master_dir}"
    echo "=========================================================="
    echo


    run_eval_stage \
        "${master_dir}" \
        "${manifest}" \
        "Base" \
        "${BASE_RUN}" \
        "0 1 2 3 4 5" \
        "${alias_prefix}_base_t0_5_10k"


    run_eval_stage \
        "${master_dir}" \
        "${manifest}" \
        "CL1" \
        "${V_CL1_RUN}" \
        "0 1 2 3 4 5 6" \
        "${alias_prefix}_cl1_t6_2k"


    run_eval_stage \
        "${master_dir}" \
        "${manifest}" \
        "CL2" \
        "${V_CL2_RUN}" \
        "0 1 2 3 4 5 6 7" \
        "${alias_prefix}_cl2_t7_2k"


    run_eval_stage \
        "${master_dir}" \
        "${manifest}" \
        "CL3" \
        "${V_CL3_RUN}" \
        "0 1 2 3 4 5 6 7 8" \
        "${alias_prefix}_cl3_t8_2k"


    run_eval_stage \
        "${master_dir}" \
        "${manifest}" \
        "CL4" \
        "${V_CL4_RUN}" \
        "0 1 2 3 4 5 6 7 8 9" \
        "${alias_prefix}_cl4_t9_2k"


    build_performance_matrix \
        "${manifest}" \
        "${master_dir}"


    compute_cl_metrics \
        "${variant}" \
        "${master_dir}"


    echo
    echo "[OK] ${variant} evaluation complete:"
    echo "     ${master_dir}"
    echo
    echo "Core SR matrix:"
    echo "     ${master_dir}/cl_performance_matrix.csv"

    if [ "${COMPUTE_CL_METRICS}" = "true" ]; then
        echo
        echo "CL metrics:"
        echo "     ${master_dir}/cl_metrics/cl_metrics_summary.csv"
    fi
}


# =============================================================================
# 16. Run one variant end-to-end
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


    # Strong final invariant/plasticity check against the SAME Base.
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
# 17. Master log
# =============================================================================

LOG_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/vlm_freeze_ablation_logs"

mkdir -p "${LOG_ROOT}"

MASTER_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/goal_vlm_freeze_ablation_${MASTER_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1


echo
echo "=========================================================="
echo " LaWAM LIBERO-Goal VLM Freeze Ablation"
echo "=========================================================="
echo "Existing Goal SeqFT Base:"
echo "  ${BASE_RUN}"
echo
echo "Variant A: strict_freeze_vlm"
echo "  mode        : ${STRICT_FREEZE_VLM_MODE}"
echo "  start stage : ${STRICT_FREEZE_VLM_START_STAGE}"
echo "  checkpoints : ${STRICT_FREEZE_VLM_RUN_ROOT}"
echo "  definition  : freeze entire Qwen VLM"
echo
echo "Variant B: freeze_backbone"
echo "  mode        : ${FREEZE_BACKBONE_MODE}"
echo "  start stage : ${FREEZE_BACKBONE_START_STAGE}"
echo "  checkpoints : ${FREEZE_BACKBONE_RUN_ROOT}"
echo "  definition  : freeze vision+LLM backbones; train vision merger"
echo
echo "Common CL training:"
echo "  suite       : libero_goal"
echo "  Base        : tasks 0-5, reuse existing 10K Base"
echo "  CL1         : task 6"
echo "  CL2         : task 7"
echo "  CL3         : task 8"
echo "  CL4         : task 9"
echo "  GPUs        : ${TRAIN_GPUS}"
echo "  batch/GPU   : ${PER_DEVICE_BATCH_SIZE}"
echo "  grad accum  : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  global batch: $((PER_DEVICE_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"
echo "  steps/stage : ${MAX_TRAIN_STEPS}"
echo "  warmup      : ${NUM_WARMUP_STEPS}"
echo
echo "Evaluation:"
echo "  policy GPU  : ${POLICY_GPU}"
echo "  sim GPU     : ${EVAL_GPU}"
echo "  workers     : ${EVAL_WORKERS}"
echo "  trials/task : ${NUM_TRIALS}"
echo "  videos      : ${SAVE_VIDEOS}"
echo
echo "CL metrics:"
echo "  enabled     : ${COMPUTE_CL_METRICS}"
if [ "${COMPUTE_CL_METRICS}" = "true" ]; then
    echo "  script      : ${CL_METRICS_SCRIPT}"
fi
echo "=========================================================="
echo


# =============================================================================
# 18. Execute variants IN ORDER
# =============================================================================

run_variant \
    "strict_freeze_vlm" \
    "${STRICT_FREEZE_VLM_MODE}" \
    "${STRICT_FREEZE_VLM_START_STAGE}" \
    "${STRICT_FREEZE_VLM_RUN_ROOT}" \
    "freeze_vlm" \
    "${STRICT_FREEZE_VLM_OUTPUT_ROOT}" \
    "freeze_vlm"


run_variant \
    "freeze_backbone" \
    "${FREEZE_BACKBONE_MODE}" \
    "${FREEZE_BACKBONE_START_STAGE}" \
    "${FREEZE_BACKBONE_RUN_ROOT}" \
    "freeze_backbone" \
    "${FREEZE_BACKBONE_OUTPUT_ROOT}" \
    "freeze_backbone"


# =============================================================================
# 19. Complete
# =============================================================================

echo
echo "=========================================================="
echo " LIBERO-Goal VLM Freeze Ablation completed"
echo "=========================================================="
echo "Existing SeqFT Base:"
echo "  ${BASE_RUN}"
echo
echo "Strict Freeze-VLM checkpoints:"
echo "  ${STRICT_FREEZE_VLM_RUN_ROOT}"
echo "Strict Freeze-VLM results:"
echo "  ${STRICT_FREEZE_VLM_OUTPUT_ROOT}"
echo
echo "Freeze-Backbone checkpoints:"
echo "  ${FREEZE_BACKBONE_RUN_ROOT}"
echo "Freeze-Backbone results:"
echo "  ${FREEZE_BACKBONE_OUTPUT_ROOT}"
echo
echo "Master log:"
echo "  ${MASTER_LOG}"
echo "=========================================================="