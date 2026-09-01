#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# LaWAM LIBERO-Goal Oracle Flow-Head Isolation
# Strict Freeze-VLM-Interface extension
#
# Purpose
# -------
# Diagnostic experiment for locating behavioral forgetting in LaWAM continual
# learning. Historical Flow heads are preserved independently, while the shared
# upstream state continues across CL stages.
#
# Historical Flow heads:
#
#   F_Base : Base tasks 0-5
#   F_CL1  : task 6
#   F_CL2  : task 7
#   F_CL3  : task 8
#   F_CL4  : task 9
#
# Training rule
# -------------
# Every CL stage starts from:
#
#   shared parameters = previous CL stage
#   Flow head         = SAME Base Flow
#
# Thus:
#
#   CL1 : (S_Base, F_Base) --T6--> (S_1, F_CL1)
#   CL2 : (S_1,    F_Base) --T7--> (S_2, F_CL2)
#   CL3 : (S_2,    F_Base) --T8--> (S_3, F_CL3)
#   CL4 : (S_3,    F_Base) --T9--> (S_4, F_CL4)
#
# F_CL1 ... F_CL4 are independently adapted from the SAME F_Base rather than
# sequentially overwriting one another.
#
# Evaluation rule
# ---------------
# At stage CLk, the current shared-stage checkpoint is combined with the Flow
# head from the task's introduction stage:
#
#   Base tasks 0-5 -> current shared + F_Base
#   task 6         -> current shared + F_CL1
#   task 7         -> current shared + F_CL2
#   task 8         -> current shared + F_CL3
#   task 9         -> current shared + F_CL4
#
# IMPORTANT COMPONENT BOUNDARY
# ----------------------------
# Only `policy_backend.flow.*` is replaced during Oracle Flow composition.
#
# For the strict `freeze_vlm_oracle_flow` variant, the following tensors are
# frozen to their Base values throughout CL1-CL4:
#
#   policy_backend.vlm.*
#   policy_backend.act_query
#   policy_backend.flow_action_query
#
# The following remain continually trainable:
#
#   policy_backend.vlm_to_lam.*     (QFormer / VLMToLAM)
#   policy_backend.lam.decoder.*    (LaWM decoder)
#
# The current stage Flow (`policy_backend.flow.*`) is trainable, but is reset to
# F_Base before each new CL stage.
#
# Variants
# --------
# A) seqft_oracle_flow
#      VLM backbone/merger trainable; embeddings frozen; queries trainable.
#
# B) freeze_backbone_oracle_flow
#      vision + LLM backbones frozen; merger and queries trainable.
#
# C) freeze_vlm_oracle_flow   [STRICT INTERFACE ISOLATION]
#      entire policy_backend.vlm frozen
#      act_query frozen
#      flow_action_query frozen
#      QFormer/VLMToLAM trainable
#      LaWM decoder trainable
#      stage-specific Flow trainable
#
# Safe defaults in this v3 script
# -------------------------------
#   SeqFT-OH            : skip
#   Freeze-Backbone-OH  : skip
#   Strict Freeze-VLM-OH: cl, start from CL1
#
# Therefore running this script without mode overrides will NOT retrain the
# already-completed SeqFT-OH / Freeze-Backbone-OH experiments.
#
# Notes
# -----
# - Only final_model is retained for every CL stage; periodic saves are disabled.
# - Composed checkpoints are temporary and deleted after use by default.
# - Each strict Freeze-VLM checkpoint is bitwise compared against the Base
#   checkpoint for VLM + act_query + flow_action_query.
# - `NCCL_TOPO_FILE`, `NCCL_GRAPH_FILE`, and related overrides are explicitly
#   cleared because this server previously exposed an invalid NCCL topology
#   override.
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
# 1. Required helper scripts
# =============================================================================

COMPOSE_SCRIPT="${ROOT}/scripts/compose_lawam_flow_run.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
CL_METRICS_SCRIPT="${ROOT}/scripts/compute_libero_cl_metrics.py"

for required in \
    "${COMPOSE_SCRIPT}" \
    "${SUMMARY_SCRIPT}" \
    "${CL_METRICS_SCRIPT}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Missing required script:"
        echo "        ${required}"
        exit 1
    fi
done


# =============================================================================
# 2. Existing formal Goal Base and new experiment roots
# =============================================================================

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

ORACLE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/oracle_flow_isolation"

SEQFT_ORACLE_RUN_ROOT="${ORACLE_ROOT}/seqft"
FREEZE_BACKBONE_ORACLE_RUN_ROOT="${ORACLE_ROOT}/freeze_backbone"
FREEZE_VLM_ORACLE_RUN_ROOT="${ORACLE_ROOT}/freeze_vlm_interface"

OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/oracle_flow_isolation"
SEQFT_ORACLE_OUTPUT_ROOT="${OUTPUT_ROOT}/seqft"
FREEZE_BACKBONE_ORACLE_OUTPUT_ROOT="${OUTPUT_ROOT}/freeze_backbone"
FREEZE_VLM_ORACLE_OUTPUT_ROOT="${OUTPUT_ROOT}/freeze_vlm_interface"

LOG_ROOT="${ORACLE_ROOT}/logs"

mkdir -p \
    "${SEQFT_ORACLE_RUN_ROOT}" \
    "${FREEZE_BACKBONE_ORACLE_RUN_ROOT}" \
    "${FREEZE_VLM_ORACLE_RUN_ROOT}" \
    "${SEQFT_ORACLE_OUTPUT_ROOT}" \
    "${FREEZE_BACKBONE_ORACLE_OUTPUT_ROOT}" \
    "${FREEZE_VLM_ORACLE_OUTPUT_ROOT}" \
    "${LOG_ROOT}"


# =============================================================================
# 3. Resource configuration
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
# 4. Formal CL hyperparameters
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

# Disable periodic stage checkpoints; final_model is still saved by the trainer.
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"

ORIGINAL_LR="${ORIGINAL_LR:-0.0001}"


# =============================================================================
# 5. Variant / runtime controls
# =============================================================================

SEQFT_ORACLE_MODE="${SEQFT_ORACLE_MODE:-skip}"
SEQFT_ORACLE_START_STAGE="${SEQFT_ORACLE_START_STAGE:-1}"

FREEZE_BACKBONE_ORACLE_MODE="${FREEZE_BACKBONE_ORACLE_MODE:-skip}"
FREEZE_BACKBONE_ORACLE_START_STAGE="${FREEZE_BACKBONE_ORACLE_START_STAGE:-1}"

FREEZE_VLM_ORACLE_MODE="${FREEZE_VLM_ORACLE_MODE:-cl}"
FREEZE_VLM_ORACLE_START_STAGE="${FREEZE_VLM_ORACLE_START_STAGE:-1}"

# Temporary composed runs can be several GB. Delete them immediately by default.
KEEP_COMPOSED_RUNS="${KEEP_COMPOSED_RUNS:-false}"

# Re-open composed checkpoints and verify that Flow exactly equals the donor.
VERIFY_COMPOSED_FLOW="${VERIFY_COMPOSED_FLOW:-true}"


validate_mode() {
    local name="$1"
    local mode="$2"
    local start_stage="$3"

    case "${mode}" in
        cl|eval|skip)
            ;;
        *)
            echo "[ERROR] ${name} must be cl, eval, or skip."
            exit 1
            ;;
    esac

    if ! [[ "${start_stage}" =~ ^[1-4]$ ]]; then
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
    "SEQFT_ORACLE_MODE" \
    "${SEQFT_ORACLE_MODE}" \
    "${SEQFT_ORACLE_START_STAGE}"

validate_mode \
    "FREEZE_BACKBONE_ORACLE_MODE" \
    "${FREEZE_BACKBONE_ORACLE_MODE}" \
    "${FREEZE_BACKBONE_ORACLE_START_STAGE}"

validate_mode \
    "FREEZE_VLM_ORACLE_MODE" \
    "${FREEZE_VLM_ORACLE_MODE}" \
    "${FREEZE_VLM_ORACLE_START_STAGE}"

validate_bool "KEEP_COMPOSED_RUNS" "${KEEP_COMPOSED_RUNS}"
validate_bool "VERIFY_COMPOSED_FLOW" "${VERIFY_COMPOSED_FLOW}"


# =============================================================================
# 6. Runtime environment
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
# 7. Generic helpers
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
        echo "[ERROR] ${name}: run not found."
        exit 1
    fi

    for required in \
        "${run}/config.yaml" \
        "${run}/dataset_statistics.json" \
        "${run}/final_model/pytorch_model.pt"
    do
        if [ ! -f "${required}" ]; then
            echo "[ERROR] ${name}: missing file:"
            echo "        ${required}"
            exit 1
        fi
    done

    echo "[OK] ${name}"
    echo "     run  : ${run}"
    echo "     ckpt : ${run}/final_model/pytorch_model.pt"
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
        raise RuntimeError(f"Missing embodiment tag: {tag}")

    for section in ("action", "state"):
        if section not in reference[tag] or section not in current[tag]:
            raise RuntimeError(f"Missing normalization section: {tag}/{section}")

        if reference[tag][section] != current[tag][section]:
            raise RuntimeError(
                f"Normalization statistics changed: {tag}/{section}"
            )

print("[OK] action/state normalization is identical to Base.")
PY
}


remove_composed_run() {
    local path="$1"

    if [ "${KEEP_COMPOSED_RUNS}" = "true" ]; then
        echo "[INFO] Keeping composed run:"
        echo "       ${path}"
        return
    fi

    if [ -n "${path}" ] && [ -d "${path}" ]; then
        echo "[cleanup] removing temporary composed run:"
        echo "          ${path}"
        rm -rf "${path}"
    fi
}


compose_run() {
    local shared_run="$1"
    local flow_run="$2"
    local output_run="$3"

    local -a args=(
        python "${COMPOSE_SCRIPT}"
        --shared-run "${shared_run}"
        --flow-run "${flow_run}"
        --output-run "${output_run}"
        --overwrite
    )

    if [ "${VERIFY_COMPOSED_FLOW}" = "true" ]; then
        args+=(--verify-output)
    fi

    "${args[@]}"
}



# =============================================================================
# 7b. Strict Freeze-VLM support preflight
# =============================================================================

verify_strict_freeze_support() {
    python - <<'PY'
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)


required_fields = {
    "freeze_vlm_all",
    "freeze_act_query",
    "freeze_flow_action_query",
}

dataclass_fields = set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
missing_fields = sorted(required_fields - dataclass_fields)

if missing_fields:
    raise RuntimeError(
        "freeze_policy.py is missing strict-freeze fields: "
        f"{missing_fields}"
    )


cfg_path = Path("starVLA/config/training/train_libero.yaml")
cfg = OmegaConf.load(cfg_path)

freeze_cfg = cfg.trainer.freeze

missing_cfg = [
    key
    for key in sorted(required_fields)
    if key not in freeze_cfg
]

if missing_cfg:
    raise RuntimeError(
        "train_libero.yaml is missing strict-freeze keys: "
        f"{missing_cfg}"
    )


print(
    "[OK] Strict Freeze-VLM support detected: "
    "freeze_vlm_all / freeze_act_query / "
    "freeze_flow_action_query."
)
PY
}


if [ "${FREEZE_VLM_ORACLE_MODE}" != "skip" ]; then
    verify_strict_freeze_support
fi


# =============================================================================
# 8. Locate existing formal Goal Base
# =============================================================================

if [ -n "${BASE_RUN:-}" ]; then
    echo "[INFO] Using explicitly supplied BASE_RUN:"
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
# 9. Per-process temporary-composition root
# =============================================================================

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TMP_COMPOSE_ROOT="${ORACLE_ROOT}/_tmp_composed_${PIPELINE_TIMESTAMP}_$$"
mkdir -p "${TMP_COMPOSE_ROOT}"


cleanup_tmp_root() {
    if [ "${KEEP_COMPOSED_RUNS}" = "true" ]; then
        echo "[INFO] KEEP_COMPOSED_RUNS=true; temporary root retained:"
        echo "       ${TMP_COMPOSE_ROOT}"
        return
    fi

    if [ -d "${TMP_COMPOSE_ROOT}" ]; then
        rm -rf "${TMP_COMPOSE_ROOT}"
    fi
}

trap cleanup_tmp_root EXIT INT TERM


# =============================================================================
# 10. Verify variant config after training
# =============================================================================

verify_variant_config() {
    local variant="$1"
    local config_path="$2"

    python - "${variant}" "${config_path}" "${ORIGINAL_LR}" <<'PY'
import math
import sys
from omegaconf import OmegaConf

variant, config_path, lr_s = sys.argv[1:4]
expected_lr = float(lr_s)

cfg = OmegaConf.load(config_path)
freeze = cfg.trainer.freeze
lr_cfg = cfg.trainer.learning_rate

if variant == "seqft_oracle_flow":
    expected = {
        "freeze_vision_backbone": False,
        "freeze_llm_backbone": False,
        "freeze_last_llm_layer": True,
        "freeze_embedding": True,
        "unfreeze_vision_merger": True,

        "freeze_vlm_all": False,
        "freeze_act_query": False,
        "freeze_flow_action_query": False,

        "keep_llm_first_n_layers": 16,
        "unfreeze_llm_last_n_layers": -1,
        "unfreeze_lam_decoder": True,
    }

elif variant == "freeze_backbone_oracle_flow":
    expected = {
        "freeze_vision_backbone": True,
        "freeze_llm_backbone": True,
        "freeze_last_llm_layer": True,
        "freeze_embedding": True,
        "unfreeze_vision_merger": True,

        "freeze_vlm_all": False,
        "freeze_act_query": False,
        "freeze_flow_action_query": False,

        "keep_llm_first_n_layers": 16,
        "unfreeze_llm_last_n_layers": -1,
        "unfreeze_lam_decoder": True,
    }

elif variant == "freeze_vlm_oracle_flow":
    expected = {
        "freeze_vision_backbone": True,
        "freeze_llm_backbone": True,
        "freeze_last_llm_layer": True,
        "freeze_embedding": True,
        "unfreeze_vision_merger": False,

        "freeze_vlm_all": True,
        "freeze_act_query": True,
        "freeze_flow_action_query": True,

        "keep_llm_first_n_layers": 16,
        "unfreeze_llm_last_n_layers": -1,
        "unfreeze_lam_decoder": True,
    }

else:
    raise RuntimeError(f"Unknown variant: {variant}")

bad = []

new_optional_flags = {
    "freeze_vlm_all",
    "freeze_act_query",
    "freeze_flow_action_query",
}

for key, value in expected.items():
    # Older SeqFT-OH / FreezeBackbone-OH checkpoints were created before these
    # three strict-isolation flags existed. Missing values are therefore
    # interpreted as their historical default False.
    default = False if key in new_optional_flags else None
    actual = freeze.get(key, default)

    if actual != value:
        bad.append((f"trainer.freeze.{key}", actual, value))

for group in ("vlm", "action_model", "world_model"):
    actual = float(lr_cfg[group].lr)
    if not math.isclose(actual, expected_lr, rel_tol=0.0, abs_tol=1e-12):
        bad.append((f"trainer.learning_rate.{group}.lr", actual, expected_lr))

load_flow = bool(cfg.trainer.get("load_pretrained_policy_flow", True))
if not load_flow:
    bad.append(("trainer.load_pretrained_policy_flow", load_flow, True))

if bad:
    text = "\n".join(
        f"  {k}: actual={a!r}, expected={e!r}"
        for k, a, e in bad
    )
    raise RuntimeError(f"{variant} config mismatch:\n{text}")

print(f"[OK] {variant} config verified.")
PY
}


# =============================================================================
# 11. Checkpoint diagnostics
# =============================================================================

report_flow_change() {
    local reference_ckpt="$1"
    local current_ckpt="$2"
    local label="$3"

    python - \
        "${reference_ckpt}" \
        "${current_ckpt}" \
        "${label}" <<'PY'

import sys
import torch

ref_path, cur_path, label = sys.argv[1:4]
prefix = "policy_backend.flow."


def load(path):
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
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)

            if (
                isinstance(nested, dict)
                and nested
                and any(torch.is_tensor(v) for v in nested.values())
            ):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Invalid checkpoint: {path}"
        )

    return obj


ref = load(ref_path)
cur = load(cur_path)

rkeys = {
    k
    for k, v in ref.items()
    if k.startswith(prefix) and torch.is_tensor(v)
}

ckeys = {
    k
    for k, v in cur.items()
    if k.startswith(prefix) and torch.is_tensor(v)
}

if not rkeys or rkeys != ckeys:
    raise RuntimeError(
        f"{label}: Flow key mismatch; "
        f"ref={len(rkeys)} cur={len(ckeys)}"
    )

changed = []

for key in sorted(rkeys):
    a = ref[key]
    b = cur[key]

    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    ):
        changed.append(key)

print(
    f"[flow-change] {label}: "
    f"checked={len(rkeys)} "
    f"changed_vs_Base={len(changed)}"
)

PY
}


verify_frozen_vlm_interface() {
    local reference_ckpt="$1"
    local current_ckpt="$2"
    local label="$3"

    python - \
        "${reference_ckpt}" \
        "${current_ckpt}" \
        "${label}" <<'PY'

import sys
import torch

base_path, current_path, label = sys.argv[1:4]


def load(path):
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
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)

            if (
                isinstance(nested, dict)
                and nested
                and any(torch.is_tensor(v) for v in nested.values())
            ):
                obj = nested
                break

    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Invalid checkpoint: {path}"
        )

    return obj


base = load(base_path)
current = load(current_path)


def is_interface_key(key):
    return (
        key.startswith("policy_backend.vlm.")
        or key == "policy_backend.act_query"
        or key.startswith("policy_backend.act_query.")
        or key == "policy_backend.flow_action_query"
        or key.startswith("policy_backend.flow_action_query.")
    )


base_keys = {
    k
    for k, v in base.items()
    if is_interface_key(k) and torch.is_tensor(v)
}

current_keys = {
    k
    for k, v in current.items()
    if is_interface_key(k) and torch.is_tensor(v)
}


if not base_keys:
    raise RuntimeError(
        "No VLM-interface tensors found in Base checkpoint."
    )


missing = sorted(base_keys - current_keys)
extra = sorted(current_keys - base_keys)

if missing or extra:
    raise RuntimeError(
        f"{label}: VLM-interface key mismatch. "
        f"missing={missing[:10]}, extra={extra[:10]}"
    )


changed = []

for key in sorted(base_keys):
    a = base[key]
    b = current[key]

    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    ):
        changed.append(key)


if changed:
    raise RuntimeError(
        f"{label}: strict VLM-interface freeze failed. "
        f"changed={len(changed)}, "
        f"examples={changed[:20]}"
    )


vlm_count = sum(
    k.startswith("policy_backend.vlm.")
    for k in base_keys
)

act_count = sum(
    k == "policy_backend.act_query"
    or k.startswith("policy_backend.act_query.")
    for k in base_keys
)

flow_q_count = sum(
    k == "policy_backend.flow_action_query"
    or k.startswith("policy_backend.flow_action_query.")
    for k in base_keys
)


print(
    f"[strict-freeze-check] {label}: "
    f"VLM={vlm_count}, "
    f"act_query={act_count}, "
    f"flow_action_query={flow_q_count}, "
    f"changed=0 exact=True"
)

PY
}

# =============================================================================
# 12. Train one oracle-flow variant
# =============================================================================

train_variant() {
    local variant="$1"
    local run_root="$2"
    local run_suffix="$3"
    local start_stage="$4"

    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES=4

    local prev_shared_run

    if [ "${start_stage}" -eq 1 ]; then
        prev_shared_run="${BASE_RUN}"
    else
        local prev_stage=$((start_stage - 1))
        local prev_task_id=$((prev_stage + 5))
        local prev_id="cl${prev_stage}_t${prev_task_id}_2k_4gpu_bs32_ga2_${run_suffix}"

        prev_shared_run=$(find_run \
            "${run_root}" \
            "*+${prev_id}"
        )

        verify_run \
            "${variant} existing CL${prev_stage}" \
            "${prev_shared_run}"

        verify_variant_config \
            "${variant}" \
            "${prev_shared_run}/config.yaml"
    fi


    for stage in $(seq "${start_stage}" 4); do
        local task_id=$((stage + 5))
        local run_id="cl${stage}_t${task_id}_2k_4gpu_bs32_ga2_${run_suffix}"

        local init_run=""
        local init_ckpt=""

        if [ "${stage}" -eq 1 ]; then
            # Base shared + Base Flow is exactly the existing Base checkpoint.
            init_run="${BASE_RUN}"
            init_ckpt="${BASE_CKPT}"
        else
            # KEY INTERVENTION:
            # previous shared state + SAME Base Flow.
            init_run="${TMP_COMPOSE_ROOT}/${variant}_train_init_cl${stage}"

            compose_run \
                "${prev_shared_run}" \
                "${BASE_RUN}" \
                "${init_run}"

            init_ckpt="${init_run}/final_model/pytorch_model.pt"
        fi

        echo
        echo "=========================================================="
        echo " ${variant}: training CL${stage}"
        echo "=========================================================="
        echo "Task                  : Goal ${task_id}"
        echo "Shared source          : ${prev_shared_run}"
        echo "Flow initialization    : Base Flow"
        echo "Initialization run     : ${init_run}"
        echo "Initialization ckpt    : ${init_ckpt}"
        echo "New run ID             : ${run_id}"
        echo "=========================================================="
        echo

        common_args=(
            "--run_root_dir=${run_root}"
            "--run_id=${run_id}"

            "--datasets.vla_data.cl_suite=libero_goal"
            "--datasets.vla_data.cl_task_ids=[${task_id}]"

            "--datasets.vla_data.use_task_filtered_statistics=false"
            "--trainer.use_pretrained_dataset_statistics=true"
            "--trainer.pretrained_checkpoint=${init_ckpt}"
            "--trainer.load_pretrained_policy_flow=true"

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
            seqft_oracle_flow)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=false"
                    "--trainer.freeze.freeze_llm_backbone=false"
                    "--trainer.freeze.unfreeze_vision_merger=true"

                    "--trainer.freeze.freeze_vlm_all=false"
                    "--trainer.freeze.freeze_act_query=false"
                    "--trainer.freeze.freeze_flow_action_query=false"
                )
                ;;

            freeze_backbone_oracle_flow)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=true"
                    "--trainer.freeze.freeze_llm_backbone=true"
                    "--trainer.freeze.unfreeze_vision_merger=true"

                    "--trainer.freeze.freeze_vlm_all=false"
                    "--trainer.freeze.freeze_act_query=false"
                    "--trainer.freeze.freeze_flow_action_query=false"
                )
                ;;

            freeze_vlm_oracle_flow)
                variant_args=(
                    "--trainer.freeze.freeze_vision_backbone=true"
                    "--trainer.freeze.freeze_llm_backbone=true"
                    "--trainer.freeze.freeze_embedding=true"
                    "--trainer.freeze.unfreeze_vision_merger=false"

                    "--trainer.freeze.freeze_vlm_all=true"
                    "--trainer.freeze.freeze_act_query=true"
                    "--trainer.freeze.freeze_flow_action_query=true"
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

        verify_run \
            "${variant} CL${stage}" \
            "${current_run}"

        verify_statistics \
            "${BASE_STATS}" \
            "${current_run}/dataset_statistics.json"

        verify_variant_config \
            "${variant}" \
            "${current_run}/config.yaml"

        if [ "${variant}" = "freeze_vlm_oracle_flow" ]; then
            verify_frozen_vlm_interface \
                "${BASE_CKPT}" \
                "${current_run}/final_model/pytorch_model.pt" \
                "${variant}/CL${stage}"
        fi

        report_flow_change \
            "${BASE_CKPT}" \
            "${current_run}/final_model/pytorch_model.pt" \
            "${variant}/CL${stage}"

        echo
        echo "[OK] ${variant} CL${stage} completed."
        echo "     Historical head F_CL${stage}:"
        echo "     ${current_run}/final_model/pytorch_model.pt"
        echo

        # The composed initializer is disposable after training finishes.
        if [ "${stage}" -gt 1 ]; then
            remove_composed_run "${init_run}"
        fi

        # CRITICAL: only the shared part of this stage is inherited next stage.
        # Its Flow will be replaced with Base Flow before the next training stage.
        prev_shared_run="${current_run}"
    done

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
}


# =============================================================================
# 13. Resolve a variant's CL1-CL4 chain
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
        if [ "${variant}" = "freeze_vlm_oracle_flow" ]; then
            verify_frozen_vlm_interface \
                "${BASE_CKPT}" \
                "${run}/final_model/pytorch_model.pt" \
                "${variant}/$(basename "${run}")"
        fi
    done
}


get_stage_run() {
    local stage="$1"

    case "${stage}" in
        Base)
            echo "${BASE_RUN}"
            ;;
        CL1)
            echo "${V_CL1_RUN}"
            ;;
        CL2)
            echo "${V_CL2_RUN}"
            ;;
        CL3)
            echo "${V_CL3_RUN}"
            ;;
        CL4)
            echo "${V_CL4_RUN}"
            ;;
        *)
            echo "[ERROR] Unknown stage: ${stage}" >&2
            exit 1
            ;;
    esac
}


get_head_donor_run() {
    local head="$1"

    case "${head}" in
        Base)
            echo "${BASE_RUN}"
            ;;
        CL1)
            echo "${V_CL1_RUN}"
            ;;
        CL2)
            echo "${V_CL2_RUN}"
            ;;
        CL3)
            echo "${V_CL3_RUN}"
            ;;
        CL4)
            echo "${V_CL4_RUN}"
            ;;
        *)
            echo "[ERROR] Unknown head: ${head}" >&2
            exit 1
            ;;
    esac
}


# =============================================================================
# 14. Evaluation environment
# =============================================================================

prepare_eval_environment() {
    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true

    export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
    export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
    export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
}


# =============================================================================
# 15. Evaluate one stage/head group
# =============================================================================

run_eval_group() {
    local variant="$1"
    local master_dir="$2"
    local manifest="$3"
    local stage="$4"
    local head_label="$5"
    local shared_run="$6"
    local flow_run="$7"
    local task_ids="$8"

    local eval_model_run=""
    local composed_run=""

    # If both sources are identical, no composition is necessary.
    if [ "${shared_run}" = "${flow_run}" ]; then
        eval_model_run="${shared_run}"
    else
        composed_run="${TMP_COMPOSE_ROOT}/eval_${variant}_${stage}_head_${head_label}"

        compose_run \
            "${shared_run}" \
            "${flow_run}" \
            "${composed_run}"

        eval_model_run="${composed_run}"
    fi

    local ckpt="${eval_model_run}/final_model/pytorch_model.pt"
    local alias="${variant}_${stage}_head_${head_label}"

    echo
    echo "=========================================================="
    echo " Oracle evaluation group"
    echo "=========================================================="
    echo "Variant       : ${variant}"
    echo "Current stage : ${stage}"
    echo "Head          : ${head_label}"
    echo "Tasks         : ${task_ids}"
    echo "Shared run    : ${shared_run}"
    echo "Flow donor    : ${flow_run}"
    echo "Eval run      : ${eval_model_run}"
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

    for required in \
        "${suite_dir}/episodes.jsonl" \
        "${suite_dir}/summary.json"
    do
        if [ ! -f "${required}" ]; then
            echo "[ERROR] Missing evaluation file:"
            echo "        ${required}"
            exit 1
        fi
    done


    read -r -a TASK_ARRAY <<< "${task_ids}"

    python "${SUMMARY_SCRIPT}" \
        --run-dir "${suite_dir}" \
        --task-ids "${TASK_ARRAY[@]}" \
        --expected-trials "${NUM_TRIALS}"


    printf \
        "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${stage}" \
        "${head_label}" \
        "${shared_run}" \
        "${flow_run}" \
        "${suite_dir}" \
        "${task_ids}" \
        >> "${manifest}"


    echo "[OK] ${stage}/${head_label} evaluation completed."

    if [ -n "${composed_run}" ]; then
        remove_composed_run "${composed_run}"
    fi
}


# =============================================================================
# 16. Build oracle-flow SR matrix
# =============================================================================

build_oracle_matrix() {
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
    manifest = list(csv.DictReader(f, delimiter="\t"))

stage_order = ["Base", "CL1", "CL2", "CL3", "CL4"]

expected_tasks = {
    "Base": set(range(0, 6)),
    "CL1": set(range(0, 7)),
    "CL2": set(range(0, 8)),
    "CL3": set(range(0, 9)),
    "CL4": set(range(0, 10)),
}

stage_results = {stage: {} for stage in stage_order}
long_rows = []

for item in manifest:
    stage = item["stage"]
    head = item["head"]
    run_dir = Path(item["eval_run_dir"])

    if stage not in stage_results:
        raise RuntimeError(f"Unexpected stage in manifest: {stage}")

    per_task_path = run_dir / "per_task_summary.json"

    with per_task_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    for row in rows:
        task_id = int(row["task_id"])
        sr = float(row["success_rate"])
        successes = int(row["successes"])
        trials = int(row["trials"])

        if trials != expected_trials:
            raise RuntimeError(
                f"{stage}/{head}/task{task_id}: "
                f"expected {expected_trials} trials, got {trials}"
            )

        if task_id in stage_results[stage]:
            raise RuntimeError(
                f"Duplicate oracle evaluation for {stage}/task{task_id}"
            )

        stage_results[stage][task_id] = sr

        long_rows.append(
            {
                "stage": stage,
                "head": head,
                "task_id": task_id,
                "task_description": row["task_description"],
                "successes": successes,
                "trials": trials,
                "success_rate": sr,
                "shared_run": item["shared_run"],
                "flow_run": item["flow_run"],
                "eval_run_dir": item["eval_run_dir"],
            }
        )


for stage in stage_order:
    actual = set(stage_results[stage])
    expected = expected_tasks[stage]

    if actual != expected:
        raise RuntimeError(
            f"{stage}: incomplete oracle matrix row; "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )


long_path = master_dir / "oracle_flow_per_task_long.csv"

with long_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stage",
            "head",
            "task_id",
            "task_description",
            "successes",
            "trials",
            "success_rate",
            "shared_run",
            "flow_run",
            "eval_run_dir",
        ],
    )
    writer.writeheader()
    writer.writerows(long_rows)


matrix_path = master_dir / "cl_performance_matrix.csv"

with matrix_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = (
        ["stage"]
        + [f"task_{i}" for i in range(10)]
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
print("Oracle Flow-Head SR Matrix")
print("Stage | " + " | ".join(f"T{i}" for i in range(10)) + " | Mean")
print("-" * 108)

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
PY
}


# =============================================================================
# 17. Evaluate a complete oracle-flow variant
# =============================================================================

evaluate_variant() {
    local variant="$1"
    local output_root="$2"

    prepare_eval_environment

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local master_dir="${output_root}/cl_full_${timestamp}"
    mkdir -p "${master_dir}"

    local manifest="${master_dir}/oracle_flow_manifest.tsv"

    printf \
        "stage\thead\tshared_run\tflow_run\teval_run_dir\ttask_ids\n" \
        > "${manifest}"


    echo
    echo "=========================================================="
    echo " ${variant}: Oracle Flow-Head evaluation"
    echo "=========================================================="
    echo "Policy GPU    : ${POLICY_GPU}"
    echo "Evaluator GPU : ${EVAL_GPU}"
    echo "Workers       : ${EVAL_WORKERS}"
    echo "Trials/task   : ${NUM_TRIALS}"
    echo "Output        : ${master_dir}"
    echo "=========================================================="
    echo


    # -------------------------------------------------------------------------
    # Base: shared Base + F_Base, tasks 0-5.
    # -------------------------------------------------------------------------
    run_eval_group \
        "${variant}" \
        "${master_dir}" \
        "${manifest}" \
        "Base" \
        "Base" \
        "${BASE_RUN}" \
        "${BASE_RUN}" \
        "0 1 2 3 4 5"


    # -------------------------------------------------------------------------
    # CL1-CL4.
    # For each stage:
    #   current shared + F_Base on tasks 0-5
    #   current shared + F_CLi on task 5+i for every learned CL task.
    # -------------------------------------------------------------------------
    for stage_idx in 1 2 3 4; do
        local stage="CL${stage_idx}"
        local shared_run
        shared_run=$(get_stage_run "${stage}")

        # Base historical Flow.
        run_eval_group \
            "${variant}" \
            "${master_dir}" \
            "${manifest}" \
            "${stage}" \
            "Base" \
            "${shared_run}" \
            "${BASE_RUN}" \
            "0 1 2 3 4 5"


        # Historical CL Flow heads.
        for head_idx in $(seq 1 "${stage_idx}"); do
            local head="CL${head_idx}"
            local donor
            donor=$(get_head_donor_run "${head}")
            local task_id=$((head_idx + 5))

            run_eval_group \
                "${variant}" \
                "${master_dir}" \
                "${manifest}" \
                "${stage}" \
                "${head}" \
                "${shared_run}" \
                "${donor}" \
                "${task_id}"
        done
    done


    build_oracle_matrix \
        "${manifest}" \
        "${master_dir}"


    python "${CL_METRICS_SCRIPT}" \
        "${master_dir}/cl_performance_matrix.csv" \
        --names "${variant}" \
        --base-tasks 0 1 2 3 4 5 \
        --cl-tasks 6 7 8 9 \
        --output-dir "${master_dir}/cl_metrics"


    cat > "${master_dir}/ORACLE_PROTOCOL.txt" <<EOF
This is an Oracle Flow-Head Isolation diagnostic, not a final task-ID-free CL policy.

Variant:
  ${variant}

Oracle-exchanged component:
  policy_backend.flow.*

Historical Flow routing:
  task 0-5 -> F_Base
  task 6   -> F_CL1
  task 7   -> F_CL2
  task 8   -> F_CL3
  task 9   -> F_CL4

All non-Flow tensors are taken from the current shared-stage checkpoint.

For freeze_vlm_oracle_flow (strict VLM-interface isolation):

  FROZEN exactly to Base throughout CL1-CL4:
    policy_backend.vlm.*
    policy_backend.act_query
    policy_backend.flow_action_query

  CONTINUALLY TRAINED:
    policy_backend.vlm_to_lam.*
    policy_backend.lam.decoder.*

  STAGE-SPECIFIC TRAINABLE / ORACLE-ROUTED:
    policy_backend.flow.*

For seqft_oracle_flow and freeze_backbone_oracle_flow, query trainability follows
their corresponding training configuration.

At evaluation stage CLk, all tasks use the current shared-stage checkpoint and
the historical Flow head from their own introduction stage.
EOF


    echo
    echo "=========================================================="
    echo " ${variant}: Oracle evaluation complete"
    echo "=========================================================="
    echo "SR matrix:"
    echo "  ${master_dir}/cl_performance_matrix.csv"
    echo "CL metrics:"
    echo "  ${master_dir}/cl_metrics/cl_metrics_summary.csv"
    echo "Detailed routing:"
    echo "  ${master_dir}/oracle_flow_per_task_long.csv"
    echo "Manifest:"
    echo "  ${manifest}"
    echo "=========================================================="
}


# =============================================================================
# 18. Run one variant end-to-end
# =============================================================================

run_variant() {
    local variant="$1"
    local mode="$2"
    local start_stage="$3"
    local run_root="$4"
    local run_suffix="$5"
    local output_root="$6"

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

    evaluate_variant \
        "${variant}" \
        "${output_root}"
}


# =============================================================================
# 19. Master log / experiment summary
# =============================================================================

MASTER_LOG="${LOG_ROOT}/oracle_flow_isolation_${PIPELINE_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1


echo
echo "=========================================================="
echo " LaWAM LIBERO-Goal Oracle Flow-Head Isolation"
echo "=========================================================="
echo "Base:"
echo "  ${BASE_RUN}"
echo
echo "Flow isolation:"
echo "  Historical head tensors : policy_backend.flow.*"
echo "  CL head init             : always F_Base"
echo "  eval routing             : oracle task-stage head"
echo "  query policy             : variant-dependent"
echo
echo "SeqFT-OracleFlow:"
echo "  mode        : ${SEQFT_ORACLE_MODE}"
echo "  start stage : ${SEQFT_ORACLE_START_STAGE}"
echo "  run root    : ${SEQFT_ORACLE_RUN_ROOT}"
echo
echo "FreezeBackbone-OracleFlow:"
echo "  mode        : ${FREEZE_BACKBONE_ORACLE_MODE}"
echo "  start stage : ${FREEZE_BACKBONE_ORACLE_START_STAGE}"
echo "  run root    : ${FREEZE_BACKBONE_ORACLE_RUN_ROOT}"
echo
echo "Strict FreezeVLM-Interface-OracleFlow:"
echo "  mode        : ${FREEZE_VLM_ORACLE_MODE}"
echo "  start stage : ${FREEZE_VLM_ORACLE_START_STAGE}"
echo "  run root    : ${FREEZE_VLM_ORACLE_RUN_ROOT}"
echo "  policy      : entire policy_backend.vlm frozen"
echo "  merger      : frozen as part of VLM"
echo "  act_query   : frozen"
echo "  flow_query  : frozen"
echo "  QFormer     : trainable"
echo "  LaWM decoder: trainable"
echo "  Flow        : current stage head trainable"
echo
echo "Training:"
echo "  GPUs        : ${TRAIN_GPUS}"
echo "  batch/GPU   : ${PER_DEVICE_BATCH_SIZE}"
echo "  grad accum  : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  global batch: $((PER_DEVICE_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"
echo "  steps/stage : ${MAX_TRAIN_STEPS}"
echo "  warmup      : ${NUM_WARMUP_STEPS}"
echo "  save int.   : ${SAVE_INTERVAL} (periodic saves disabled by default)"
echo
echo "Evaluation:"
echo "  policy GPU  : ${POLICY_GPU}"
echo "  sim GPU     : ${EVAL_GPU}"
echo "  workers     : ${EVAL_WORKERS}"
echo "  trials/task : ${NUM_TRIALS}"
echo
echo "Temporary compositions:"
echo "  root        : ${TMP_COMPOSE_ROOT}"
echo "  keep        : ${KEEP_COMPOSED_RUNS}"
echo "  verify Flow : ${VERIFY_COMPOSED_FLOW}"
echo "=========================================================="
echo


# =============================================================================
# 20. Execute variants
# =============================================================================

run_variant \
    "seqft_oracle_flow" \
    "${SEQFT_ORACLE_MODE}" \
    "${SEQFT_ORACLE_START_STAGE}" \
    "${SEQFT_ORACLE_RUN_ROOT}" \
    "oracle_flow_seqft" \
    "${SEQFT_ORACLE_OUTPUT_ROOT}"


run_variant \
    "freeze_backbone_oracle_flow" \
    "${FREEZE_BACKBONE_ORACLE_MODE}" \
    "${FREEZE_BACKBONE_ORACLE_START_STAGE}" \
    "${FREEZE_BACKBONE_ORACLE_RUN_ROOT}" \
    "oracle_flow_freeze_backbone" \
    "${FREEZE_BACKBONE_ORACLE_OUTPUT_ROOT}"


run_variant \
    "freeze_vlm_oracle_flow" \
    "${FREEZE_VLM_ORACLE_MODE}" \
    "${FREEZE_VLM_ORACLE_START_STAGE}" \
    "${FREEZE_VLM_ORACLE_RUN_ROOT}" \
    "oracle_flow_freeze_vlm_interface" \
    "${FREEZE_VLM_ORACLE_OUTPUT_ROOT}"


# =============================================================================
# 21. Complete
# =============================================================================

echo
echo "=========================================================="
echo " Oracle Flow-Head Isolation completed"
echo "=========================================================="
echo "SeqFT-OracleFlow checkpoints:"
echo "  ${SEQFT_ORACLE_RUN_ROOT}"
echo "SeqFT-OracleFlow results:"
echo "  ${SEQFT_ORACLE_OUTPUT_ROOT}"
echo
echo "FreezeBackbone-OracleFlow checkpoints:"
echo "  ${FREEZE_BACKBONE_ORACLE_RUN_ROOT}"
echo "FreezeBackbone-OracleFlow results:"
echo "  ${FREEZE_BACKBONE_ORACLE_OUTPUT_ROOT}"
echo
echo "Strict FreezeVLM-Interface-OracleFlow checkpoints:"
echo "  ${FREEZE_VLM_ORACLE_RUN_ROOT}"
echo "Strict FreezeVLM-Interface-OracleFlow results:"
echo "  ${FREEZE_VLM_ORACLE_OUTPUT_ROOT}"
echo
echo "Master log:"
echo "  ${MASTER_LOG}"
echo "=========================================================="