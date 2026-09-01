#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# LaWAM LIBERO-Goal
# Frozen-Upstream + Task-Specific Flow-Head Oracle
#
# Scientific question
# -------------------
# With the ENTIRE Base upstream representation fixed, how well can each
# subsequent LIBERO-Goal task be learned by adapting ONLY a new Flow action head?
#
# Frozen exactly to Base:
#   policy_backend.vlm.*
#   policy_backend.act_query
#   policy_backend.flow_action_query
#   policy_backend.vlm_to_lam.*
#   policy_backend.lam.*
#   every other non-Flow policy tensor
#
# Trainable:
#   policy_backend.flow.*
#
# Training protocol
# -----------------
# Every CL task is trained INDEPENDENTLY from the SAME Base checkpoint:
#
#   Base -> T6 -> F_CL1
#   Base -> T7 -> F_CL2
#   Base -> T8 -> F_CL3
#   Base -> T9 -> F_CL4
#
# There is NO sequential inheritance between CL1/CL2/CL3/CL4.
#
# Evaluation protocol
# -------------------
# Oracle routing:
#
#   tasks 0-5 -> original Base checkpoint / F_Base
#   task 6    -> independently trained CL1 Flow checkpoint
#   task 7    -> independently trained CL2 Flow checkpoint
#   task 8    -> independently trained CL3 Flow checkpoint
#   task 9    -> independently trained CL4 Flow checkpoint
#
# Because every non-Flow tensor is bitwise identical to Base, historical
# forgetting should be structurally eliminated.  HOWEVER, for a rigorous
# empirical sanity check, we DO NOT copy/reuse historical SR values.
#
# Instead, every visible task at every CL stage is independently re-evaluated
# with a fresh full rollout batch (NUM_TRIALS per task) under oracle routing:
#
#   Base : Base -> T0-T5
#   CL1  : Base -> T0-T5, F6 -> T6
#   CL2  : Base -> T0-T5, F6 -> T6, F7 -> T7
#   CL3  : Base -> T0-T5, F6 -> T6, F7 -> T7, F8 -> T8
#   CL4  : Base -> T0-T5, F6 -> T6, F7 -> T7, F8 -> T8, F9 -> T9
#
# This yields 6+7+8+9+10 = 40 empirical task-stage evaluations.  Small SR
# differences across repeated evaluations reflect rollout sampling noise rather
# than parameter forgetting, since the routed checkpoint is unchanged.
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

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
CL_METRICS_SCRIPT="${ROOT}/scripts/compute_libero_cl_metrics.py"

for required in \
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
# 2. Paths
# =============================================================================

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_head_only_oracle"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"

OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_head_only_oracle"

LOG_ROOT="${EXPERIMENT_ROOT}/logs"

mkdir -p \
    "${RUN_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${LOG_ROOT}"


# =============================================================================
# 3. Resource configuration
# =============================================================================

TRAIN_GPUS="${TRAIN_GPUS:-0,2,7}"

POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"

EVAL_WORKERS="${EVAL_WORKERS:-16}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"

if [ "${NUM_TRAIN_GPUS}" -lt 1 ]; then
    echo "[ERROR] At least one training GPU is required."
    echo "        TRAIN_GPUS=${TRAIN_GPUS}"
    exit 1
fi


# =============================================================================
# 4. Training hyperparameters
# =============================================================================

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-28}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"

NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"

LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"

# Disable periodic checkpoints. final_model is still saved by train_starvla.py.
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"

ORIGINAL_LR="${ORIGINAL_LR:-0.0001}"


# =============================================================================
# 5. Runtime controls
# =============================================================================

# cl   : train missing/requested heads, then evaluate
# eval : evaluate existing CL1-CL4 heads only
FLOW_HEAD_ONLY_MODE="${FLOW_HEAD_ONLY_MODE:-cl}"

# Because all heads are independent, START_STAGE only controls which head is
# trained first.  CL3 does not depend on CL1 or CL2.
FLOW_HEAD_ONLY_START_STAGE="${FLOW_HEAD_ONLY_START_STAGE:-1}"


case "${FLOW_HEAD_ONLY_MODE}" in
    cl|eval)
        ;;
    *)
        echo "[ERROR] FLOW_HEAD_ONLY_MODE must be cl or eval."
        exit 1
        ;;
esac

if ! [[ "${FLOW_HEAD_ONLY_START_STAGE}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] FLOW_HEAD_ONLY_START_STAGE must be 1, 2, 3, or 4."
    exit 1
fi


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

# The server previously exposed an invalid injected NCCL topology file.
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

    python - \
        "${reference_stats}" \
        "${current_stats}" <<'PY'
import json
import sys

reference_path, current_path = sys.argv[1:3]

with open(reference_path, "r", encoding="utf-8") as f:
    reference = json.load(f)

with open(current_path, "r", encoding="utf-8") as f:
    current = json.load(f)

for tag in reference:
    if tag not in current:
        raise RuntimeError(
            f"Missing embodiment tag: {tag}"
        )

    for section in ("action", "state"):
        if (
            section not in reference[tag]
            or section not in current[tag]
        ):
            raise RuntimeError(
                f"Missing normalization section: "
                f"{tag}/{section}"
            )

        if (
            reference[tag][section]
            != current[tag][section]
        ):
            raise RuntimeError(
                "Normalization statistics changed: "
                f"{tag}/{section}"
            )

print(
    "[OK] action/state normalization "
    "is identical to Base."
)
PY
}


# =============================================================================
# 8. Preflight: ensure the code supports train_flow_only
# =============================================================================

verify_flow_only_support() {
    python - <<'PY'
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)


fields = set(
    LatentWorldPolicyFreezeConfig.__dataclass_fields__
)

if "train_flow_only" not in fields:
    raise RuntimeError(
        "freeze_policy.py does not contain "
        "`train_flow_only`. Replace it with the "
        "Flow-only version first."
    )


cfg_path = Path(
    "starVLA/config/training/train_libero.yaml"
)

cfg = OmegaConf.load(cfg_path)
freeze_cfg = cfg.trainer.freeze

if "train_flow_only" not in freeze_cfg:
    raise RuntimeError(
        "train_libero.yaml is missing "
        "`trainer.freeze.train_flow_only`."
    )


print(
    "[OK] train_flow_only support detected "
    "in freeze_policy.py and train_libero.yaml."
)
PY
}


verify_flow_only_support


# =============================================================================
# 9. Locate the formal Goal Base
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


verify_run \
    "Existing Goal SeqFT Base" \
    "${BASE_RUN}"

BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"


# =============================================================================
# 10. Config verification
# =============================================================================

verify_flow_only_config() {
    local config_path="$1"
    local expected_task_id="$2"

    python - \
        "${config_path}" \
        "${expected_task_id}" \
        "${BASE_CKPT}" \
        "${ORIGINAL_LR}" <<'PY'
import math
import sys

from omegaconf import OmegaConf


config_path, task_s, base_ckpt, lr_s = sys.argv[1:5]

task_id = int(task_s)
expected_lr = float(lr_s)

cfg = OmegaConf.load(config_path)

freeze = cfg.trainer.freeze

if not bool(
    freeze.get(
        "train_flow_only",
        False,
    )
):
    raise RuntimeError(
        "trainer.freeze.train_flow_only is not True."
    )


load_flow = bool(
    cfg.trainer.get(
        "load_pretrained_policy_flow",
        True,
    )
)

if not load_flow:
    raise RuntimeError(
        "load_pretrained_policy_flow must be True so "
        "every task starts from the SAME Base Flow."
    )


actual_ckpt = str(
    cfg.trainer.pretrained_checkpoint
)

if actual_ckpt != str(base_ckpt):
    raise RuntimeError(
        "Flow-only head did not initialize from Base. "
        f"actual={actual_ckpt}, expected={base_ckpt}"
    )


task_ids = list(
    cfg.datasets.vla_data.cl_task_ids
)

if task_ids != [task_id]:
    raise RuntimeError(
        "Unexpected CL task filter: "
        f"actual={task_ids}, expected={[task_id]}"
    )


action_lr = float(
    cfg.trainer.learning_rate.action_model.lr
)

if not math.isclose(
    action_lr,
    expected_lr,
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise RuntimeError(
        "Unexpected Flow learning rate: "
        f"{action_lr} != {expected_lr}"
    )


print(
    f"[OK] Flow-only config verified "
    f"for task {task_id}."
)
PY
}


# =============================================================================
# 11. HARD checkpoint audit (v3):
#
#     The Flow/action module is serialized under TWO namespaces in this repo:
#       1) policy_backend.flow.*      (canonical policy path)
#       2) policy_action_head.*       (checkpoint alias / action-head path)
#
#     Both namespaces are ALLOWED to change during Flow-only training.
#     EVERY OTHER tensor must remain bitwise identical to the formal Base.
#
#     This fixes the v2 false positive where policy_action_head.* was
#     incorrectly classified as frozen upstream.
# =============================================================================

verify_only_flow_changed() {
    local current_ckpt="$1"
    local label="$2"

    python - \
        "${BASE_CKPT}" \
        "${current_ckpt}" \
        "${label}" <<'PY'
import sys
import torch


base_path, current_path, label = sys.argv[1:4]


def load(path):
    kwargs = {"map_location": "cpu"}

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
        raise RuntimeError(f"Invalid checkpoint: {path}")

    return obj


def tensor_changed(a, b):
    return (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    )


def is_canonical_flow_key(key):
    return (
        key == "policy_backend.flow"
        or key.startswith("policy_backend.flow.")
    )


def is_flow_alias_key(key):
    # LaWAM/StarVLA serializes the trainable Flow/action-head module through
    # this additional state-dict namespace as well.
    return (
        key == "policy_action_head"
        or key.startswith("policy_action_head.")
    )


base = load(base_path)
current = load(current_path)

base_tensor_keys = {
    key for key, value in base.items() if torch.is_tensor(value)
}
current_tensor_keys = {
    key for key, value in current.items() if torch.is_tensor(value)
}

if base_tensor_keys != current_tensor_keys:
    missing = sorted(base_tensor_keys - current_tensor_keys)
    extra = sorted(current_tensor_keys - base_tensor_keys)
    raise RuntimeError(
        f"{label}: checkpoint tensor-key mismatch. "
        f"missing={missing[:20]}, extra={extra[:20]}"
    )

canonical_flow_keys = sorted(
    key for key in base_tensor_keys if is_canonical_flow_key(key)
)
flow_alias_keys = sorted(
    key for key in base_tensor_keys if is_flow_alias_key(key)
)
allowed_flow_keys = set(canonical_flow_keys) | set(flow_alias_keys)
upstream_keys = sorted(base_tensor_keys - allowed_flow_keys)

if not canonical_flow_keys:
    raise RuntimeError(
        f"{label}: no policy_backend.flow.* tensors found. "
        "Cannot verify Flow-only isolation."
    )

changed_canonical_flow = [
    key for key in canonical_flow_keys
    if tensor_changed(base[key], current[key])
]
changed_flow_alias = [
    key for key in flow_alias_keys
    if tensor_changed(base[key], current[key])
]
changed_upstream = [
    key for key in upstream_keys
    if tensor_changed(base[key], current[key])
]

# Critical scientific requirement: frozen upstream must be bitwise identical.
if changed_upstream:
    raise RuntimeError(
        f"{label}: FLOW-ONLY ISOLATION FAILED. "
        f"Frozen upstream tensors changed: count={len(changed_upstream)}, "
        f"examples={changed_upstream[:30]}"
    )

# At least one allowed Flow/action-head tensor must actually update.
changed_allowed_total = (
    len(changed_canonical_flow) + len(changed_flow_alias)
)
if changed_allowed_total == 0:
    raise RuntimeError(
        f"{label}: neither policy_backend.flow.* nor "
        "policy_action_head.* changed from Base. "
        "Training may not have updated the Flow/action head."
    )

print(
    f"[flow-only-check-v3] {label}: "
    f"upstream_checked={len(upstream_keys)}, "
    "upstream_changed=0, "
    f"canonical_flow_checked={len(canonical_flow_keys)}, "
    f"canonical_flow_changed={len(changed_canonical_flow)}, "
    f"flow_alias_checked={len(flow_alias_keys)}, "
    f"flow_alias_changed={len(changed_flow_alias)}, "
    "exact_upstream=True"
)

if flow_alias_keys:
    print(
        f"[flow-only-check-v3] {label}: detected checkpoint Flow alias "
        "namespace policy_action_head.* (allowed to change)."
    )
else:
    print(
        f"[flow-only-check-v3] {label}: no policy_action_head.* alias "
        "present; canonical Flow namespace is sufficient."
    )
PY
}


# =============================================================================
# 12. Train four independent Flow heads
# =============================================================================

train_flow_heads() {
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES="${NUM_TRAIN_GPUS}"

    for stage in $(seq "${FLOW_HEAD_ONLY_START_STAGE}" 4); do
        local task_id=$((stage + 5))

        local run_id="cl${stage}_t${task_id}_2k_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_flow_head_only_from_base"

        echo
        echo "=========================================================="
        echo " Flow-Head-Only: training CL${stage}"
        echo "=========================================================="
        echo "Task               : Goal ${task_id}"
        echo "Initialization     : SAME formal Base checkpoint"
        echo "Base checkpoint    : ${BASE_CKPT}"
        echo "Trainable          : policy_backend.flow.* ONLY"
        echo "Sequential inherit : NONE"
        echo "Run ID             : ${run_id}"
        echo "=========================================================="
        echo

        bash train_lawam.sh \
            "--run_root_dir=${RUN_ROOT}" \
            "--run_id=${run_id}" \
            \
            "--datasets.vla_data.cl_suite=libero_goal" \
            "--datasets.vla_data.cl_task_ids=[${task_id}]" \
            \
            "--datasets.vla_data.use_task_filtered_statistics=false" \
            "--trainer.use_pretrained_dataset_statistics=true" \
            "--trainer.pretrained_checkpoint=${BASE_CKPT}" \
            "--trainer.load_pretrained_policy_flow=true" \
            \
            "--trainer.freeze.train_flow_only=true" \
            "--trainer.freeze.unfreeze_lam_decoder=false" \
            \
            "--trainer.learning_rate.vlm.lr=${ORIGINAL_LR}" \
            "--trainer.learning_rate.action_model.lr=${ORIGINAL_LR}" \
            "--trainer.learning_rate.world_model.lr=${ORIGINAL_LR}" \
            \
            "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
            "--datasets.vla_data.num_workers=${NUM_WORKERS}" \
            "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}" \
            "--datasets.vla_data.persistent_workers=true" \
            \
            "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
            "--trainer.max_train_steps=${MAX_TRAIN_STEPS}" \
            "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}" \
            \
            "--trainer.logging_frequency=${LOGGING_FREQUENCY}" \
            "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}" \
            "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}" \
            "--trainer.save_interval=${SAVE_INTERVAL}"

        local current_run

        current_run=$(find_run \
            "${RUN_ROOT}" \
            "*+${run_id}"
        )

        verify_run \
            "Flow-only CL${stage}" \
            "${current_run}"

        verify_statistics \
            "${BASE_STATS}" \
            "${current_run}/dataset_statistics.json"

        verify_flow_only_config \
            "${current_run}/config.yaml" \
            "${task_id}"

        verify_only_flow_changed \
            "${current_run}/final_model/pytorch_model.pt" \
            "CL${stage}/T${task_id}"

        echo
        echo "[OK] Independent Flow head CL${stage}/T${task_id} completed."
        echo "     ${current_run}"
        echo
    done

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
}


# =============================================================================
# 13. Resolve all four trained heads
# =============================================================================

resolve_head_runs() {
    CL1_RUN=$(find_run \
        "${RUN_ROOT}" \
        "*+cl1_t6_2k_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_flow_head_only_from_base"
    )

    CL2_RUN=$(find_run \
        "${RUN_ROOT}" \
        "*+cl2_t7_2k_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_flow_head_only_from_base"
    )

    CL3_RUN=$(find_run \
        "${RUN_ROOT}" \
        "*+cl3_t8_2k_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_flow_head_only_from_base"
    )

    CL4_RUN=$(find_run \
        "${RUN_ROOT}" \
        "*+cl4_t9_2k_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_flow_head_only_from_base"
    )


    verify_run "Flow-only CL1/T6" "${CL1_RUN}"
    verify_run "Flow-only CL2/T7" "${CL2_RUN}"
    verify_run "Flow-only CL3/T8" "${CL3_RUN}"
    verify_run "Flow-only CL4/T9" "${CL4_RUN}"


    verify_flow_only_config \
        "${CL1_RUN}/config.yaml" \
        "6"

    verify_flow_only_config \
        "${CL2_RUN}/config.yaml" \
        "7"

    verify_flow_only_config \
        "${CL3_RUN}/config.yaml" \
        "8"

    verify_flow_only_config \
        "${CL4_RUN}/config.yaml" \
        "9"


    verify_only_flow_changed \
        "${CL1_RUN}/final_model/pytorch_model.pt" \
        "CL1/T6"

    verify_only_flow_changed \
        "${CL2_RUN}/final_model/pytorch_model.pt" \
        "CL2/T7"

    verify_only_flow_changed \
        "${CL3_RUN}/final_model/pytorch_model.pt" \
        "CL3/T8"

    verify_only_flow_changed \
        "${CL4_RUN}/final_model/pytorch_model.pt" \
        "CL4/T9"
}


get_head_run() {
    local head="$1"

    case "${head}" in
        Base)
            echo "${BASE_RUN}"
            ;;
        CL1)
            echo "${CL1_RUN}"
            ;;
        CL2)
            echo "${CL2_RUN}"
            ;;
        CL3)
            echo "${CL3_RUN}"
            ;;
        CL4)
            echo "${CL4_RUN}"
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
# 15. Evaluate one routed checkpoint on one task block for one CL stage
# =============================================================================

run_eval_block() {
    local master_dir="$1"
    local manifest="$2"
    local stage_label="$3"
    local head_label="$4"
    local model_run="$5"
    local task_ids="$6"

    local ckpt="${model_run}/final_model/pytorch_model.pt"
    local alias="flow_head_only_${stage_label}_${head_label}"

    echo
    echo "=========================================================="
    echo " Full empirical Flow-Head-Only evaluation"
    echo "=========================================================="
    echo "Stage      : ${stage_label}"
    echo "Routed head: ${head_label}"
    echo "Tasks      : ${task_ids}"
    echo "Model run  : ${model_run}"
    echo "Checkpoint : ${ckpt}"
    echo "Trials/task: ${NUM_TRIALS}"
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
        "%s\t%s\t%s\t%s\t%s\n" \
        "${stage_label}" \
        "${head_label}" \
        "${model_run}" \
        "${suite_dir}" \
        "${task_ids}" \
        >> "${manifest}"

    echo "[OK] ${stage_label}/${head_label} evaluation completed."
}


# =============================================================================
# 16. Evaluate one COMPLETE CL stage under oracle routing
# =============================================================================

evaluate_one_stage() {
    local master_dir="$1"
    local manifest="$2"
    local stage_label="$3"
    local stage_num="$4"

    echo
    echo "##########################################################"
    echo " Empirical CL stage: ${stage_label}"
    echo "##########################################################"

    run_eval_block \
        "${master_dir}" \
        "${manifest}" \
        "${stage_label}" \
        "Base" \
        "${BASE_RUN}" \
        "0 1 2 3 4 5"

    if [ "${stage_num}" -gt 0 ]; then
        for head_stage in $(seq 1 "${stage_num}"); do
            local task_id=$((head_stage + 5))
            local head_label="CL${head_stage}"
            local head_run

            head_run=$(get_head_run "${head_label}")

            run_eval_block \
                "${master_dir}" \
                "${manifest}" \
                "${stage_label}" \
                "${head_label}" \
                "${head_run}" \
                "${task_id}"
        done
    fi
}


# =============================================================================
# 17. Build a FULLY EMPIRICAL CL matrix (NO copied/reused SR values)
# =============================================================================

build_full_empirical_matrix() {
    local manifest="$1"
    local master_dir="$2"

    python - \
        "${manifest}" \
        "${master_dir}" \
        "${NUM_TRIALS}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
master_dir = Path(sys.argv[2])
expected_trials = int(sys.argv[3])

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = list(csv.DictReader(f, delimiter="\t"))

stage_order = ["Base", "CL1", "CL2", "CL3", "CL4"]
stage_index = {stage: i for i, stage in enumerate(stage_order)}

expected_stage_tasks = {
    "Base": set(range(0, 6)),
    "CL1": set(range(0, 7)),
    "CL2": set(range(0, 8)),
    "CL3": set(range(0, 9)),
    "CL4": set(range(0, 10)),
}

expected_head_tasks = {
    "Base": set(range(0, 6)),
    "CL1": {6},
    "CL2": {7},
    "CL3": {8},
    "CL4": {9},
}

stage_results = defaultdict(dict)
long_rows = []

for item in manifest:
    stage = item["stage"]
    head = item["head"]
    run_dir = Path(item["eval_run_dir"])

    if stage not in expected_stage_tasks:
        raise RuntimeError(f"Unexpected stage in manifest: {stage}")
    if head not in expected_head_tasks:
        raise RuntimeError(f"Unexpected head in manifest: {head}")

    if head != "Base":
        head_num = int(head[2:])
        if head_num > stage_index[stage]:
            raise RuntimeError(
                f"Invalid oracle routing: stage={stage}, head={head}"
            )

    per_task_path = run_dir / "per_task_summary.json"
    if not per_task_path.is_file():
        raise RuntimeError(f"Missing summary: {per_task_path}")

    with per_task_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    actual_tasks = {int(row["task_id"]) for row in rows}
    if actual_tasks != expected_head_tasks[head]:
        raise RuntimeError(
            f"{stage}/{head}: routed task mismatch; "
            f"expected={sorted(expected_head_tasks[head])}, "
            f"actual={sorted(actual_tasks)}"
        )

    for row in rows:
        task_id = int(row["task_id"])
        sr = float(row["success_rate"])
        successes = int(row["successes"])
        trials = int(row["trials"])

        if trials != expected_trials:
            raise RuntimeError(
                f"{stage}/{head}/task{task_id}: expected "
                f"{expected_trials} trials, got {trials}"
            )

        if task_id in stage_results[stage]:
            raise RuntimeError(
                f"Duplicate empirical value for {stage}/task{task_id}"
            )

        stage_results[stage][task_id] = sr

        long_rows.append(
            {
                "stage": stage,
                "routed_head": head,
                "task_id": task_id,
                "task_description": row["task_description"],
                "successes": successes,
                "trials": trials,
                "success_rate": sr,
                "model_run": item["model_run"],
                "eval_run_dir": item["eval_run_dir"],
            }
        )

for stage in stage_order:
    actual = set(stage_results[stage])
    expected = expected_stage_tasks[stage]
    if actual != expected:
        raise RuntimeError(
            f"{stage}: incomplete empirical matrix row; "
            f"expected={sorted(expected)}, actual={sorted(actual)}"
        )

matrix_path = master_dir / "cl_performance_matrix.csv"
with matrix_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["stage"] + [f"task_{i}" for i in range(10)] + ["mean_seen_task_sr"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for stage in stage_order:
        values = stage_results[stage]
        row = {"stage": stage}
        for task_id in range(10):
            row[f"task_{task_id}"] = (
                f"{values[task_id]:.4f}" if task_id in values else ""
            )
        row["mean_seen_task_sr"] = f"{sum(values.values()) / len(values):.4f}"
        writer.writerow(row)

long_path = master_dir / "full_empirical_per_task.csv"
with long_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "stage",
        "routed_head",
        "task_id",
        "task_description",
        "successes",
        "trials",
        "success_rate",
        "model_run",
        "eval_run_dir",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(
        sorted(
            long_rows,
            key=lambda r: (stage_index[r["stage"]], r["task_id"]),
        )
    )

first_stage_for_task = {
    0: "Base", 1: "Base", 2: "Base", 3: "Base", 4: "Base", 5: "Base",
    6: "CL1", 7: "CL2", 8: "CL3", 9: "CL4",
}

stability_rows = []
for task_id in range(10):
    reference_stage = first_stage_for_task[task_id]
    reference_sr = stage_results[reference_stage][task_id]
    reference_idx = stage_index[reference_stage]

    for later_stage in stage_order[reference_idx + 1:]:
        if task_id not in stage_results[later_stage]:
            continue

        later_sr = stage_results[later_stage][task_id]
        delta = later_sr - reference_sr
        forgetting_drop = max(0.0, reference_sr - later_sr)

        stability_rows.append(
            {
                "task_id": task_id,
                "reference_stage": reference_stage,
                "later_stage": later_stage,
                "reference_sr": reference_sr,
                "later_sr": later_sr,
                "delta_later_minus_reference": delta,
                "absolute_delta": abs(delta),
                "forgetting_drop": forgetting_drop,
            }
        )

stability_path = master_dir / "historical_stability_deltas.csv"
with stability_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "task_id",
        "reference_stage",
        "later_stage",
        "reference_sr",
        "later_sr",
        "delta_later_minus_reference",
        "absolute_delta",
        "forgetting_drop",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stability_rows)

summary_rows = []
for task_id in range(10):
    rows = [r for r in stability_rows if r["task_id"] == task_id]
    ref_stage = first_stage_for_task[task_id]
    ref_sr = stage_results[ref_stage][task_id]

    if rows:
        summary_rows.append(
            {
                "task_id": task_id,
                "reference_stage": ref_stage,
                "reference_sr": ref_sr,
                "num_later_reevals": len(rows),
                "max_absolute_delta": max(r["absolute_delta"] for r in rows),
                "max_forgetting_drop": max(r["forgetting_drop"] for r in rows),
                "mean_delta": sum(r["delta_later_minus_reference"] for r in rows) / len(rows),
            }
        )
    else:
        summary_rows.append(
            {
                "task_id": task_id,
                "reference_stage": ref_stage,
                "reference_sr": ref_sr,
                "num_later_reevals": 0,
                "max_absolute_delta": 0.0,
                "max_forgetting_drop": 0.0,
                "mean_delta": 0.0,
            }
        )

stability_summary_path = master_dir / "historical_stability_summary.csv"
with stability_summary_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "task_id",
        "reference_stage",
        "reference_sr",
        "num_later_reevals",
        "max_absolute_delta",
        "max_forgetting_drop",
        "mean_delta",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print()
print("Frozen-Upstream + Task-Specific Flow: FULL EMPIRICAL SR Matrix")
print("Stage | " + " | ".join(f"T{i}" for i in range(10)) + " | Mean")
print("-" * 108)

for stage in stage_order:
    values = stage_results[stage]
    display = [f"{values[i]:.2f}" if i in values else "-" for i in range(10)]
    mean_sr = sum(values.values()) / len(values)
    print(
        f"{stage:>4s} | "
        + " | ".join(f"{value:>4s}" for value in display)
        + f" | {mean_sr:.4f}"
    )

print()
print("Empirical historical-stability check:")
for row in summary_rows:
    if row["num_later_reevals"] == 0:
        continue
    print(
        f"  T{row['task_id']}: "
        f"ref={row['reference_sr']:.3f}, "
        f"later_reevals={row['num_later_reevals']}, "
        f"max|delta|={row['max_absolute_delta']:.3f}, "
        f"max_drop={row['max_forgetting_drop']:.3f}"
    )

print()
print("[IMPORTANT] No historical SR value was copied or reused.")
print("            Every populated matrix cell comes from a fresh rollout batch.")
print()
print("[OK] Saved:")
print(matrix_path)
print(long_path)
print(stability_path)
print(stability_summary_path)
PY
}


# =============================================================================
# 18. Evaluate ALL 40 task-stage cells empirically
# =============================================================================

evaluate_flow_heads_full_matrix() {
    prepare_eval_environment

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local master_dir="${OUTPUT_ROOT}/cl_full_empirical_${timestamp}"
    mkdir -p "${master_dir}"

    local manifest="${master_dir}/full_eval_manifest.tsv"

    printf \
        "stage\thead\tmodel_run\teval_run_dir\ttask_ids\n" \
        > "${manifest}"

    echo
    echo "=========================================================="
    echo " Frozen-Upstream + Task-Specific Flow FULL evaluation"
    echo "=========================================================="
    echo "Policy GPU      : ${POLICY_GPU}"
    echo "Evaluator GPU   : ${EVAL_GPU}"
    echo "Workers         : ${EVAL_WORKERS}"
    echo "Trials/task     : ${NUM_TRIALS}"
    echo "Empirical cells : 40 (= 6+7+8+9+10)"
    echo "SR reuse        : DISABLED"
    echo "Output          : ${master_dir}"
    echo "=========================================================="
    echo

    evaluate_one_stage "${master_dir}" "${manifest}" "Base" 0
    evaluate_one_stage "${master_dir}" "${manifest}" "CL1" 1
    evaluate_one_stage "${master_dir}" "${manifest}" "CL2" 2
    evaluate_one_stage "${master_dir}" "${manifest}" "CL3" 3
    evaluate_one_stage "${master_dir}" "${manifest}" "CL4" 4

    build_full_empirical_matrix \
        "${manifest}" \
        "${master_dir}"

    python "${CL_METRICS_SCRIPT}" \
        "${master_dir}/cl_performance_matrix.csv" \
        --names "flow_head_only_oracle_full_empirical" \
        --base-tasks 0 1 2 3 4 5 \
        --cl-tasks 6 7 8 9 \
        --output-dir "${master_dir}/cl_metrics"

    cat > "${master_dir}/PROTOCOL.txt" <<EOF
Frozen-Upstream + Task-Specific Flow-Head Oracle diagnostic.
FULL EMPIRICAL RE-EVALUATION protocol (v3 alias-aware checkpoint audit).

Base checkpoint:
  ${BASE_RUN}

Training:
  CL1/T6 -> initialized from Base; train policy_backend.flow.* only
  CL2/T7 -> initialized from Base; train policy_backend.flow.* only
  CL3/T8 -> initialized from Base; train policy_backend.flow.* only
  CL4/T9 -> initialized from Base; train policy_backend.flow.* only

No CL stage inherits any trainable/shared parameter from a previous CL stage.
Every non-Flow checkpoint tensor is verified bitwise identical to Base.

Oracle task routing:
  tasks 0-5 -> Base checkpoint / F_Base
  task 6    -> CL1 Flow-only checkpoint / F6
  task 7    -> CL2 Flow-only checkpoint / F7
  task 8    -> CL3 Flow-only checkpoint / F8
  task 9    -> CL4 Flow-only checkpoint / F9

FULL evaluation matrix:
  Base -> Base:T0-T5
  CL1  -> Base:T0-T5 + F6:T6
  CL2  -> Base:T0-T5 + F6:T6 + F7:T7
  CL3  -> Base:T0-T5 + F6:T6 + F7:T7 + F8:T8
  CL4  -> Base:T0-T5 + F6:T6 + F7:T7 + F8:T8 + F9:T9

Each populated task-stage cell is evaluated independently with ${NUM_TRIALS}
rollouts. NO historical success-rate value is copied/reused across stages.
Total empirical task-stage cells: 40.
Total intended rollouts: $((40 * NUM_TRIALS)).

Interpretation:
  1. Current-task SR / FWT measures Flow-only plasticity under a fixed Base
     upstream representation.
  2. Repeated historical-task evaluations provide an empirical zero-forgetting
     sanity check. Because the exact routed checkpoints are unchanged, small SR
     fluctuations should be interpreted as rollout sampling noise.
  3. historical_stability_deltas.csv and historical_stability_summary.csv
     directly quantify those repeated-evaluation fluctuations.
EOF

    echo
    echo "=========================================================="
    echo " Full empirical Flow-Head-Only evaluation complete"
    echo "=========================================================="
    echo "SR matrix:"
    echo "  ${master_dir}/cl_performance_matrix.csv"
    echo "CL metrics:"
    echo "  ${master_dir}/cl_metrics/cl_metrics_summary.csv"
    echo "All empirical cells:"
    echo "  ${master_dir}/full_empirical_per_task.csv"
    echo "Historical stability deltas:"
    echo "  ${master_dir}/historical_stability_deltas.csv"
    echo "Historical stability summary:"
    echo "  ${master_dir}/historical_stability_summary.csv"
    echo "Protocol:"
    echo "  ${master_dir}/PROTOCOL.txt"
    echo "=========================================================="
}


# =============================================================================
# 19. Master log
# =============================================================================

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/flow_head_only_full_empirical_${PIPELINE_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1


echo
echo "=========================================================="
echo " LaWAM Goal: Frozen-Upstream + Flow-Head-Only Oracle"
echo " FULL EMPIRICAL RE-EVALUATION (v3)"
echo "=========================================================="
echo "Base:"
echo "  ${BASE_RUN}"
echo
echo "Mode:"
echo "  ${FLOW_HEAD_ONLY_MODE}"
echo "Start stage:"
echo "  ${FLOW_HEAD_ONLY_START_STAGE}"
echo
echo "Frozen:"
echo "  ALL non-Flow policy parameters"
echo "Trainable:"
echo "  policy_backend.flow.* ONLY"
echo
echo "Training:"
echo "  GPUs        : ${TRAIN_GPUS}"
echo "  batch/GPU   : ${PER_DEVICE_BATCH_SIZE}"
echo "  grad accum  : ${GRADIENT_ACCUMULATION_STEPS}"
echo "  num train GPUs: ${NUM_TRAIN_GPUS}"
echo "  global batch: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_GPUS))"
echo "  steps/head  : ${MAX_TRAIN_STEPS}"
echo "  warmup      : ${NUM_WARMUP_STEPS}"
echo "  save int.   : ${SAVE_INTERVAL}"
echo
echo "Evaluation:"
echo "  policy GPU       : ${POLICY_GPU}"
echo "  sim GPU          : ${EVAL_GPU}"
echo "  workers          : ${EVAL_WORKERS}"
echo "  trials/task      : ${NUM_TRIALS}"
echo "  empirical cells  : 40"
echo "  intended rollouts: $((40 * NUM_TRIALS))"
echo "  SR reuse         : disabled"
echo "  checkpoint audit : v3 alias-aware (policy_backend.flow.* + policy_action_head.*)"
echo "=========================================================="
echo


# =============================================================================
# 20. Run
# =============================================================================

if [ "${FLOW_HEAD_ONLY_MODE}" = "cl" ]; then
    train_flow_heads
fi

resolve_head_runs

evaluate_flow_heads_full_matrix


echo
echo "=========================================================="
echo " Frozen-Upstream + Flow-Head-Only experiment complete."
echo "=========================================================="
echo "Checkpoints:"
echo "  ${RUN_ROOT}"
echo
echo "Results:"
echo "  ${OUTPUT_ROOT}"
echo
echo "Master log:"
echo "  ${MASTER_LOG}"
echo "=========================================================="