#!/usr/bin/env bash

set -euo pipefail


# ==========================================================
# 0. Environment
# ==========================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh

conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

cd /home/jincai_guo/tianqi/CVPR2027/LaWAM


# ==========================================================
# 1. Experiment configuration
# ==========================================================

RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"

mkdir -p "${RUN_ROOT}"


# ----------------------------------------------------------
# Physical GPUs.
#
# Default:
#   GPU 4,5,6,7
#
# Can be overridden when launching:
#   GPUS="0,1,4,5" bash ...
# ----------------------------------------------------------

GPUS="${GPUS:-4,5,6,7}"

export CUDA_VISIBLE_DEVICES="${GPUS}"

# This experiment is designed for exactly four GPUs.
export NUM_PROCESSES=4


# ----------------------------------------------------------
# Runtime environment
# ----------------------------------------------------------

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN

unset NCCL_TOPO_FILE
unset NCCL_GRAPH_FILE
unset NCCL_CONF_FILE
unset HFAI_NCCL_OPT_LEVEL


# ==========================================================
# 2. CL training hyperparameters
# ==========================================================

PER_DEVICE_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=2

MAX_TRAIN_STEPS=2000
NUM_WARMUP_STEPS=120

NUM_WORKERS=4
VAL_NUM_WORKERS=2

LOGGING_FREQUENCY=100

EVAL_INTERVAL=500
EVAL_BATCHES=20

SAVE_INTERVAL=500


# ==========================================================
# 3. Helper: find latest run by run_id
# ==========================================================

find_latest_run() {
    local run_id="$1"

    find "${RUN_ROOT}" \
        -maxdepth 1 \
        -type d \
        -name "*+${run_id}" \
        | sort \
        | tail -n 1
}


# ==========================================================
# 4. Locate formal Base checkpoint
#
# Base:
#   LIBERO-object task 0-5
#   10K optimizer steps
# ==========================================================

if [ -n "${BASE_RUN:-}" ]; then

    echo "[INFO] Using explicitly provided BASE_RUN:"
    echo "       ${BASE_RUN}"

else

    BASE_RUN=$(find \
        "${RUN_ROOT}" \
        -maxdepth 1 \
        -type d \
        -name '*+base_t0_5_10k_4gpu_bs32_ga2' \
        | sort \
        | tail -n 1
    )

fi


if [ -z "${BASE_RUN}" ]; then
    echo "[ERROR] Formal Base run was not found."
    echo
    echo "Expected something like:"
    echo "${RUN_ROOT}/<timestamp>+base_t0_5_10k_4gpu_bs32_ga2"
    exit 1
fi


BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"


if [ ! -f "${BASE_CKPT}" ]; then
    echo "[ERROR] Base checkpoint does not exist:"
    echo "        ${BASE_CKPT}"
    exit 1
fi


if [ ! -f "${BASE_STATS}" ]; then
    echo "[ERROR] Base dataset_statistics.json does not exist:"
    echo "        ${BASE_STATS}"
    exit 1
fi


# ==========================================================
# 5. Check GPU count
# ==========================================================

IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"

if [ "${#GPU_ARRAY[@]}" -ne 4 ]; then
    echo "[ERROR] This script expects exactly 4 GPUs."
    echo "        GPUS=${GPUS}"
    exit 1
fi


# ==========================================================
# 6. Create chain-level log
# ==========================================================

CHAIN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CHAIN_LOG="${RUN_ROOT}/cl1_4_chain_${CHAIN_TIMESTAMP}.log"

echo "[INFO] Chain log:"
echo "       ${CHAIN_LOG}"

exec > >(tee -a "${CHAIN_LOG}") 2>&1


# ==========================================================
# 7. Helper: verify normalization statistics
#
# Important:
# dataset_statistics.json also contains dataset-level counts,
# e.g.
#   num_transitions
#   num_trajectories
#
# Those SHOULD change between CL stages.
#
# We only require action/state normalization to remain
# exactly identical to Base.
# ==========================================================

verify_statistics() {

    local reference_stats="$1"
    local current_stats="$2"

    python - "${reference_stats}" "${current_stats}" <<'PY'
import json
import sys

reference_path = sys.argv[1]
current_path = sys.argv[2]

with open(reference_path, "r") as f:
    reference = json.load(f)

with open(current_path, "r") as f:
    current = json.load(f)


for tag in reference:

    if tag not in current:
        raise RuntimeError(
            f"Missing embodiment tag in current statistics: {tag}"
        )

    for section in ["action", "state"]:

        if section not in reference[tag]:
            raise RuntimeError(
                f"Missing reference statistics section: "
                f"{tag}/{section}"
            )

        if section not in current[tag]:
            raise RuntimeError(
                f"Missing current statistics section: "
                f"{tag}/{section}"
            )

        if reference[tag][section] != current[tag][section]:

            raise RuntimeError(
                f"Normalization statistics changed: "
                f"{tag}/{section}"
            )


print(
    "[OK] action/state normalization is identical "
    "to Base statistics."
)
PY
}


# ==========================================================
# 8. Print experiment summary
# ==========================================================

echo
echo "=========================================================="
echo " LaWAM LIBERO-object Sequential Fine-Tuning"
echo "=========================================================="
echo
echo "Base tasks       : object 0-5"
echo "CL1              : object 6"
echo "CL2              : object 7"
echo "CL3              : object 8"
echo "CL4              : object 9"
echo
echo "RUN_ROOT          : ${RUN_ROOT}"
echo "GPUS              : ${GPUS}"
echo
echo "BASE_RUN          : ${BASE_RUN}"
echo "BASE_CKPT         : ${BASE_CKPT}"
echo "BASE_STATS        : ${BASE_STATS}"
echo
echo "per-device batch  : ${PER_DEVICE_BATCH_SIZE}"
echo "gradient accum.   : ${GRADIENT_ACCUMULATION_STEPS}"
echo "global batch      : 256"
echo
echo "steps / CL stage  : ${MAX_TRAIN_STEPS}"
echo "warmup / stage    : ${NUM_WARMUP_STEPS}"
echo "save interval     : ${SAVE_INTERVAL}"
echo "=========================================================="
echo


# ==========================================================
# 9. Optional resume point
#
# Default:
#   START_STAGE=1
#
# Examples:
#
# Start normally:
#   START_STAGE=1
#
# If CL1 already finished:
#   START_STAGE=2
#
# If CL1-CL3 already finished:
#   START_STAGE=4
# ==========================================================

START_STAGE="${START_STAGE:-1}"


if ! [[ "${START_STAGE}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] START_STAGE must be one of:"
    echo "        1, 2, 3, 4"
    exit 1
fi


# ----------------------------------------------------------
# Determine previous checkpoint when resuming.
# ----------------------------------------------------------

if [ "${START_STAGE}" -eq 1 ]; then

    PREV_RUN="${BASE_RUN}"

else

    PREV_STAGE=$((START_STAGE - 1))
    PREV_TASK_ID=$((PREV_STAGE + 5))

    PREV_RUN_ID="cl${PREV_STAGE}_t${PREV_TASK_ID}_2k_4gpu_bs32_ga2"

    PREV_RUN=$(find_latest_run "${PREV_RUN_ID}")

    if [ -z "${PREV_RUN}" ]; then
        echo "[ERROR] Cannot resume from CL${START_STAGE}."
        echo "        Previous run was not found:"
        echo "        ${PREV_RUN_ID}"
        exit 1
    fi

    echo
    echo "[INFO] Resuming sequential CL."
    echo "       START_STAGE=${START_STAGE}"
    echo "       Previous run=${PREV_RUN}"
    echo

fi


# ==========================================================
# 10. Sequential CL training
#
# STAGE 1 -> task 6
# STAGE 2 -> task 7
# STAGE 3 -> task 8
# STAGE 4 -> task 9
# ==========================================================

for STAGE in $(seq "${START_STAGE}" 4); do

    TASK_ID=$((STAGE + 5))

    RUN_ID="cl${STAGE}_t${TASK_ID}_2k_4gpu_bs32_ga2"

    PREV_CKPT="${PREV_RUN}/final_model/pytorch_model.pt"
    PREV_STATS="${PREV_RUN}/dataset_statistics.json"


    # ------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------

    if [ ! -f "${PREV_CKPT}" ]; then
        echo "[ERROR] Previous checkpoint missing:"
        echo "        ${PREV_CKPT}"
        exit 1
    fi

    if [ ! -f "${PREV_STATS}" ]; then
        echo "[ERROR] Previous statistics missing:"
        echo "        ${PREV_STATS}"
        exit 1
    fi


    echo
    echo
    echo "=========================================================="
    echo " Starting CL${STAGE}"
    echo "=========================================================="
    echo "Stage             : CL${STAGE}"
    echo "LIBERO object task  : ${TASK_ID}"
    echo
    echo "Previous run      : ${PREV_RUN}"
    echo "Previous ckpt     : ${PREV_CKPT}"
    echo "Previous stats    : ${PREV_STATS}"
    echo
    echo "New run ID        : ${RUN_ID}"
    echo "=========================================================="
    echo


    # ------------------------------------------------------
    # Formal sequential fine-tuning
    #
    # IMPORTANT:
    #
    # use_task_filtered_statistics=false
    #   -> do NOT compute statistics from the new task.
    #
    # use_pretrained_dataset_statistics=true
    #   -> inherit statistics from the previous stage.
    #
    # pretrained_checkpoint
    #   -> inherit model parameters from previous stage.
    # ------------------------------------------------------

    bash train_lawam.sh \
        --run_root_dir="${RUN_ROOT}" \
        --run_id="${RUN_ID}" \
        \
        --datasets.vla_data.cl_suite=libero_object \
        "--datasets.vla_data.cl_task_ids=[${TASK_ID}]" \
        \
        --datasets.vla_data.use_task_filtered_statistics=false \
        --trainer.use_pretrained_dataset_statistics=true \
        --trainer.pretrained_checkpoint="${PREV_CKPT}" \
        \
        --datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
        --datasets.vla_data.num_workers="${NUM_WORKERS}" \
        --datasets.vla_data.val_num_workers="${VAL_NUM_WORKERS}" \
        --datasets.vla_data.persistent_workers=true \
        \
        --trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
        --trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
        --trainer.num_warmup_steps="${NUM_WARMUP_STEPS}" \
        \
        --trainer.logging_frequency="${LOGGING_FREQUENCY}" \
        --trainer.eval_interval="${EVAL_INTERVAL}" \
        --trainer.eval_batches="${EVAL_BATCHES}" \
        --trainer.save_interval="${SAVE_INTERVAL}"


    # ======================================================
    # 11. Resolve current stage output
    # ======================================================

    CURRENT_RUN=$(find_latest_run "${RUN_ID}")


    if [ -z "${CURRENT_RUN}" ]; then
        echo "[ERROR] Output directory not found after CL${STAGE}."
        echo "        run_id=${RUN_ID}"
        exit 1
    fi


    CURRENT_CKPT="${CURRENT_RUN}/final_model/pytorch_model.pt"
    CURRENT_STATS="${CURRENT_RUN}/dataset_statistics.json"


    if [ ! -f "${CURRENT_CKPT}" ]; then
        echo "[ERROR] CL${STAGE} final checkpoint missing:"
        echo "        ${CURRENT_CKPT}"
        exit 1
    fi


    if [ ! -f "${CURRENT_STATS}" ]; then
        echo "[ERROR] CL${STAGE} statistics missing:"
        echo "        ${CURRENT_STATS}"
        exit 1
    fi


    # ======================================================
    # 12. Verify fixed Base normalization
    # ======================================================

    echo
    echo "[INFO] Verifying normalization after CL${STAGE}..."

    verify_statistics \
        "${BASE_STATS}" \
        "${CURRENT_STATS}"


    # ======================================================
    # 13. Stage summary
    # ======================================================

    echo
    echo "----------------------------------------------------------"
    echo "[OK] CL${STAGE} completed successfully."
    echo
    echo "Task:"
    echo "  object ${TASK_ID}"
    echo
    echo "Output run:"
    echo "  ${CURRENT_RUN}"
    echo
    echo "Final checkpoint:"
    echo "  ${CURRENT_CKPT}"
    echo
    echo "Statistics:"
    echo "  ${CURRENT_STATS}"
    echo "----------------------------------------------------------"
    echo


    # ======================================================
    # 14. Chain current stage -> next stage
    # ======================================================

    PREV_RUN="${CURRENT_RUN}"

done


# ==========================================================
# 15. Final summary
# ==========================================================

echo
echo
echo "=========================================================="
echo " LaWAM object CL1 -> CL4 completed successfully"
echo "=========================================================="
echo
echo "Base:"
echo "  ${BASE_RUN}"
echo
echo "Final CL4 run:"
echo "  ${PREV_RUN}"
echo
echo "Final CL4 checkpoint:"
echo "  ${PREV_RUN}/final_model/pytorch_model.pt"
echo
echo "All action/state statistics remained identical to Base."
echo
echo "Chain log:"
echo "  ${CHAIN_LOG}"
echo
echo "=========================================================="

