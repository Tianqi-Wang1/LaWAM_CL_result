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
# forgetting is structurally eliminated.  The main quantity of interest is
# therefore new-task learnability / FWT under a fixed upstream representation.
#
# To avoid injecting rollout noise into a matrix whose parameters are literally
# unchanged across later stages, each head is evaluated ONCE.  The same measured
# SR is then reused in every later CL row in which that head remains active.
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
# 4. Training hyperparameters
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
# 11. HARD checkpoint audit:
#     ALL non-Flow tensors must remain bitwise identical to Base.
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
    kwargs = {
        "map_location": "cpu",
    }

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
            obj = torch.load(
                path,
                **kwargs,
            )

    if isinstance(obj, dict):
        for wrapper in (
            "state_dict",
            "model",
            "module",
        ):
            nested = obj.get(wrapper)

            if (
                isinstance(nested, dict)
                and nested
                and any(
                    torch.is_tensor(v)
                    for v in nested.values()
                )
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


base_tensor_keys = {
    key
    for key, value in base.items()
    if torch.is_tensor(value)
}

current_tensor_keys = {
    key
    for key, value in current.items()
    if torch.is_tensor(value)
}


if base_tensor_keys != current_tensor_keys:
    missing = sorted(
        base_tensor_keys - current_tensor_keys
    )
    extra = sorted(
        current_tensor_keys - base_tensor_keys
    )

    raise RuntimeError(
        f"{label}: checkpoint tensor-key mismatch. "
        f"missing={missing[:20]}, "
        f"extra={extra[:20]}"
    )


def is_flow_key(key):
    return (
        key == "policy_backend.flow"
        or key.startswith(
            "policy_backend.flow."
        )
    )


flow_keys = sorted(
    key
    for key in base_tensor_keys
    if is_flow_key(key)
)

nonflow_keys = sorted(
    key
    for key in base_tensor_keys
    if not is_flow_key(key)
)


if not flow_keys:
    raise RuntimeError(
        f"{label}: no policy_backend.flow.* "
        "tensors found."
    )


changed_nonflow = []
changed_flow = []


for key in nonflow_keys:
    a = base[key]
    b = current[key]

    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    ):
        changed_nonflow.append(key)


for key in flow_keys:
    a = base[key]
    b = current[key]

    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    ):
        changed_flow.append(key)


if changed_nonflow:
    raise RuntimeError(
        f"{label}: FLOW-ONLY ISOLATION FAILED. "
        f"Non-Flow tensors changed: "
        f"count={len(changed_nonflow)}, "
        f"examples={changed_nonflow[:30]}"
    )


if not changed_flow:
    raise RuntimeError(
        f"{label}: Flow did not change from Base. "
        "Training may not have updated the action head."
    )


print(
    f"[flow-only-check] {label}: "
    f"nonflow_checked={len(nonflow_keys)}, "
    "nonflow_changed=0, "
    f"flow_checked={len(flow_keys)}, "
    f"flow_changed={len(changed_flow)}, "
    "exact_upstream=True"
)
PY
}


# =============================================================================
# 12. Train four independent Flow heads
# =============================================================================

train_flow_heads() {
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES=4

    for stage in $(seq "${FLOW_HEAD_ONLY_START_STAGE}" 4); do
        local task_id=$((stage + 5))

        local run_id="cl${stage}_t${task_id}_2k_4gpu_bs32_ga2_flow_head_only_from_base"

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
        '*+cl1_t6_2k_4gpu_bs32_ga2_flow_head_only_from_base'
    )

    CL2_RUN=$(find_run \
        "${RUN_ROOT}" \
        '*+cl2_t7_2k_4gpu_bs32_ga2_flow_head_only_from_base'
    )

    CL3_RUN=$(find_run \
        "${RUN_ROOT}" \
        '*+cl3_t8_2k_4gpu_bs32_ga2_flow_head_only_from_base'
    )

    CL4_RUN=$(find_run \
        "${RUN_ROOT}" \
        '*+cl4_t9_2k_4gpu_bs32_ga2_flow_head_only_from_base'
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
# 15. Evaluate one fixed head once
# =============================================================================

run_eval_head() {
    local master_dir="$1"
    local manifest="$2"
    local head_label="$3"
    local model_run="$4"
    local task_ids="$5"

    local ckpt="${model_run}/final_model/pytorch_model.pt"
    local alias="flow_head_only_${head_label}"

    echo
    echo "=========================================================="
    echo " Flow-Head-Only evaluation"
    echo "=========================================================="
    echo "Head       : ${head_label}"
    echo "Tasks      : ${task_ids}"
    echo "Model run  : ${model_run}"
    echo "Checkpoint : ${ckpt}"
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
        "%s\t%s\t%s\t%s\n" \
        "${head_label}" \
        "${model_run}" \
        "${suite_dir}" \
        "${task_ids}" \
        >> "${manifest}"


    echo "[OK] ${head_label} evaluation completed."
}


# =============================================================================
# 16. Build the CL matrix from fixed-head measurements
# =============================================================================

build_flow_only_matrix() {
    local manifest="$1"
    local master_dir="$2"

    python - \
        "${manifest}" \
        "${master_dir}" \
        "${NUM_TRIALS}" <<'PY'
import csv
import json
import sys
from pathlib import Path


manifest_path = Path(sys.argv[1])
master_dir = Path(sys.argv[2])
expected_trials = int(sys.argv[3])


with manifest_path.open(
    "r",
    encoding="utf-8",
) as f:
    manifest = list(
        csv.DictReader(
            f,
            delimiter="\t",
        )
    )


expected_head_tasks = {
    "Base": set(range(0, 6)),
    "CL1": {6},
    "CL2": {7},
    "CL3": {8},
    "CL4": {9},
}


head_results = {}
long_rows = []


for item in manifest:
    head = item["head"]
    run_dir = Path(
        item["eval_run_dir"]
    )

    if head not in expected_head_tasks:
        raise RuntimeError(
            f"Unexpected head: {head}"
        )

    per_task_path = (
        run_dir
        / "per_task_summary.json"
    )

    with per_task_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = json.load(f)

    result = {}

    for row in rows:
        task_id = int(
            row["task_id"]
        )
        sr = float(
            row["success_rate"]
        )
        successes = int(
            row["successes"]
        )
        trials = int(
            row["trials"]
        )

        if trials != expected_trials:
            raise RuntimeError(
                f"{head}/task{task_id}: "
                f"expected {expected_trials} "
                f"trials, got {trials}"
            )

        result[task_id] = sr

        long_rows.append(
            {
                "head": head,
                "task_id": task_id,
                "task_description": row[
                    "task_description"
                ],
                "successes": successes,
                "trials": trials,
                "success_rate": sr,
                "model_run": item[
                    "model_run"
                ],
                "eval_run_dir": item[
                    "eval_run_dir"
                ],
            }
        )

    actual = set(result)
    expected = expected_head_tasks[head]

    if actual != expected:
        raise RuntimeError(
            f"{head}: task mismatch; "
            f"expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )

    head_results[head] = result


if set(head_results) != set(expected_head_tasks):
    raise RuntimeError(
        "Not all five heads were evaluated. "
        f"found={sorted(head_results)}"
    )


# One canonical SR per task/head.
canonical = {}

for task_id, sr in head_results["Base"].items():
    canonical[task_id] = sr

canonical[6] = head_results["CL1"][6]
canonical[7] = head_results["CL2"][7]
canonical[8] = head_results["CL3"][8]
canonical[9] = head_results["CL4"][9]


stage_order = [
    "Base",
    "CL1",
    "CL2",
    "CL3",
    "CL4",
]

max_seen_task = {
    "Base": 5,
    "CL1": 6,
    "CL2": 7,
    "CL3": 8,
    "CL4": 9,
}


matrix_path = (
    master_dir
    / "cl_performance_matrix.csv"
)


with matrix_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    fieldnames = (
        ["stage"]
        + [
            f"task_{i}"
            for i in range(10)
        ]
        + ["mean_seen_task_sr"]
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for stage in stage_order:
        max_task = max_seen_task[stage]

        seen = {
            task_id: canonical[task_id]
            for task_id in range(
                max_task + 1
            )
        }

        row = {
            "stage": stage,
        }

        for task_id in range(10):
            row[f"task_{task_id}"] = (
                f"{seen[task_id]:.4f}"
                if task_id in seen
                else ""
            )

        row["mean_seen_task_sr"] = (
            f"{sum(seen.values()) / len(seen):.4f}"
        )

        writer.writerow(row)


long_path = (
    master_dir
    / "flow_head_only_per_task.csv"
)


with long_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "head",
            "task_id",
            "task_description",
            "successes",
            "trials",
            "success_rate",
            "model_run",
            "eval_run_dir",
        ],
    )

    writer.writeheader()
    writer.writerows(long_rows)


print()
print(
    "Frozen-Upstream + "
    "Task-Specific Flow SR Matrix"
)

print(
    "Stage | "
    + " | ".join(
        f"T{i}"
        for i in range(10)
    )
    + " | Mean"
)

print("-" * 108)


for stage in stage_order:
    max_task = max_seen_task[stage]

    seen = {
        task_id: canonical[task_id]
        for task_id in range(
            max_task + 1
        )
    }

    values = [
        (
            f"{seen[i]:.2f}"
            if i in seen
            else "-"
        )
        for i in range(10)
    ]

    mean_sr = (
        sum(seen.values())
        / len(seen)
    )

    print(
        f"{stage:>4s} | "
        + " | ".join(
            f"{value:>4s}"
            for value in values
        )
        + f" | {mean_sr:.4f}"
    )


print()
print(
    "[INFO] Historical SRs are intentionally "
    "reused across later rows because BOTH the "
    "upstream checkpoint and the routed Flow head "
    "are unchanged by construction."
)

print()
print("[OK] Saved:")
print(matrix_path)
print(long_path)
PY
}


# =============================================================================
# 17. Evaluate all five fixed heads
# =============================================================================

evaluate_flow_heads() {
    prepare_eval_environment

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")

    local master_dir="${OUTPUT_ROOT}/cl_full_${timestamp}"

    mkdir -p "${master_dir}"

    local manifest="${master_dir}/flow_head_manifest.tsv"

    printf \
        "head\tmodel_run\teval_run_dir\ttask_ids\n" \
        > "${manifest}"


    echo
    echo "=========================================================="
    echo " Frozen-Upstream + Task-Specific Flow evaluation"
    echo "=========================================================="
    echo "Policy GPU    : ${POLICY_GPU}"
    echo "Evaluator GPU : ${EVAL_GPU}"
    echo "Workers       : ${EVAL_WORKERS}"
    echo "Trials/task   : ${NUM_TRIALS}"
    echo "Output        : ${master_dir}"
    echo "=========================================================="
    echo


    # Base checkpoint is evaluated once for Base tasks 0-5.
    run_eval_head \
        "${master_dir}" \
        "${manifest}" \
        "Base" \
        "${BASE_RUN}" \
        "0 1 2 3 4 5"


    # Every CL Flow head is evaluated once on its own task.
    run_eval_head \
        "${master_dir}" \
        "${manifest}" \
        "CL1" \
        "${CL1_RUN}" \
        "6"

    run_eval_head \
        "${master_dir}" \
        "${manifest}" \
        "CL2" \
        "${CL2_RUN}" \
        "7"

    run_eval_head \
        "${master_dir}" \
        "${manifest}" \
        "CL3" \
        "${CL3_RUN}" \
        "8"

    run_eval_head \
        "${master_dir}" \
        "${manifest}" \
        "CL4" \
        "${CL4_RUN}" \
        "9"


    build_flow_only_matrix \
        "${manifest}" \
        "${master_dir}"


    python "${CL_METRICS_SCRIPT}" \
        "${master_dir}/cl_performance_matrix.csv" \
        --names "flow_head_only_oracle" \
        --base-tasks 0 1 2 3 4 5 \
        --cl-tasks 6 7 8 9 \
        --output-dir "${master_dir}/cl_metrics"


    cat > "${master_dir}/PROTOCOL.txt" <<EOF
Frozen-Upstream + Task-Specific Flow-Head Oracle diagnostic.

Base checkpoint:
  ${BASE_RUN}

Training:
  CL1/T6 -> initialized from Base; train policy_backend.flow.* only
  CL2/T7 -> initialized from Base; train policy_backend.flow.* only
  CL3/T8 -> initialized from Base; train policy_backend.flow.* only
  CL4/T9 -> initialized from Base; train policy_backend.flow.* only

No CL stage inherits any trainable/shared parameter from a previous CL stage.

Frozen exactly to Base:
  every non-Flow checkpoint tensor.

Task routing:
  tasks 0-5 -> Base checkpoint / F_Base
  task 6    -> CL1 Flow-only checkpoint
  task 7    -> CL2 Flow-only checkpoint
  task 8    -> CL3 Flow-only checkpoint
  task 9    -> CL4 Flow-only checkpoint

Each head is evaluated once.  Its measured SR is reused in all later matrix rows
because both the upstream parameters and that task's Flow parameters are
identical across those rows.

Primary interpretation:
  - current-task SR / FWT measures how much plasticity is available in Flow
    alone under the fixed Base representation.
  - historical forgetting should be structurally zero apart from evaluation
    noise, which is avoided here by reusing the same measured SR.
EOF


    echo
    echo "=========================================================="
    echo " Flow-Head-Only evaluation complete"
    echo "=========================================================="
    echo "SR matrix:"
    echo "  ${master_dir}/cl_performance_matrix.csv"
    echo "CL metrics:"
    echo "  ${master_dir}/cl_metrics/cl_metrics_summary.csv"
    echo "Per-head task results:"
    echo "  ${master_dir}/flow_head_only_per_task.csv"
    echo "Protocol:"
    echo "  ${master_dir}/PROTOCOL.txt"
    echo "=========================================================="
}


# =============================================================================
# 18. Master log
# =============================================================================

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MASTER_LOG="${LOG_ROOT}/flow_head_only_${PIPELINE_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1


echo
echo "=========================================================="
echo " LaWAM Goal: Frozen-Upstream + Flow-Head-Only Oracle"
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
echo "  global batch: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 4))"
echo "  steps/head  : ${MAX_TRAIN_STEPS}"
echo "  warmup      : ${NUM_WARMUP_STEPS}"
echo "  save int.   : ${SAVE_INTERVAL}"
echo
echo "Evaluation:"
echo "  policy GPU  : ${POLICY_GPU}"
echo "  sim GPU     : ${EVAL_GPU}"
echo "  workers     : ${EVAL_WORKERS}"
echo "  trials/task : ${NUM_TRIALS}"
echo "=========================================================="
echo


# =============================================================================
# 19. Run
# =============================================================================

if [ "${FLOW_HEAD_ONLY_MODE}" = "cl" ]; then
    train_flow_heads
fi

resolve_head_runs

evaluate_flow_heads


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