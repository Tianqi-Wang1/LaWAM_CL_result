#!/usr/bin/env bash

set -euo pipefail


# ==========================================================
# 0. Environment
# ==========================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh

conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"


# ==========================================================
# 1. Paths
# ==========================================================

RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"

PRETRAIN_CKPT="${ROOT}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt"

CL_SCRIPT="${ROOT}/scripts/train_libero_object_seqft_cl1_4.sh"

EVAL_SCRIPT="${ROOT}/scripts/eval_libero_object_seqft_all.sh"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"

mkdir -p "${RUN_ROOT}"


# ==========================================================
# 2. Resource configuration
#
# Training:
#   four GPUs
#
# Evaluation:
#   one policy GPU + one evaluator GPU
# ==========================================================

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"

POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"

EVAL_WORKERS="${EVAL_WORKERS:-16}"
NUM_TRIALS="${NUM_TRIALS:-50}"

# Full evaluation contains:
#
# Base 300
# CL1  350
# CL2  400
# CL3  450
# CL4  500
# --------
#      2000 episodes
#
# Default: do not save all videos.
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"


# ==========================================================
# 3. Resume control
#
# START_FROM:
#
#   base
#       Base -> CL1 -> CL2 -> CL3 -> CL4 -> Eval
#
#   cl
#       Existing Base -> CL stages -> Eval
#
#   eval
#       Existing Base/CL1/CL2/CL3/CL4 -> Eval only
#
# When START_FROM=cl:
#
#   CL_START_STAGE=1  -> start CL1
#   CL_START_STAGE=2  -> start CL2
#   CL_START_STAGE=3  -> start CL3
#   CL_START_STAGE=4  -> start CL4
# ==========================================================

START_FROM="${START_FROM:-base}"
CL_START_STAGE="${CL_START_STAGE:-1}"


case "${START_FROM}" in
    base|cl|eval)
        ;;
    *)
        echo "[ERROR] START_FROM must be:"
        echo "        base, cl, or eval"
        exit 1
        ;;
esac


if ! [[ "${CL_START_STAGE}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] CL_START_STAGE must be 1, 2, 3, or 4."
    exit 1
fi


if [ "${START_FROM}" = "base" ] \
   && [ "${CL_START_STAGE}" -ne 1 ]; then

    echo "[ERROR] START_FROM=base requires CL_START_STAGE=1."
    exit 1
fi


# ==========================================================
# 4. Runtime environment
# ==========================================================

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
# 5. Sanity checks
# ==========================================================

for file in \
    "${PRETRAIN_CKPT}" \
    "${CL_SCRIPT}" \
    "${EVAL_SCRIPT}" \
    "${SUMMARY_SCRIPT}"
do

    if [ ! -f "${file}" ]; then
        echo "[ERROR] Required file not found:"
        echo "        ${file}"
        exit 1
    fi

done


IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"

if [ "${#TRAIN_GPU_ARRAY[@]}" -ne 4 ]; then
    echo "[ERROR] Exactly four training GPUs are required."
    echo "        TRAIN_GPUS=${TRAIN_GPUS}"
    exit 1
fi


# ==========================================================
# 6. Helper: find latest formal run
# ==========================================================

find_latest_run() {

    local pattern="$1"

    find "${RUN_ROOT}" \
        -maxdepth 1 \
        -type d \
        -name "${pattern}" \
        | sort \
        | tail -n 1
}


# ==========================================================
# 7. Helper: verify run
# ==========================================================

verify_run() {

    local name="$1"
    local run="$2"

    if [ -z "${run}" ]; then
        echo "[ERROR] ${name} run was not found."
        exit 1
    fi

    local ckpt="${run}/final_model/pytorch_model.pt"
    local stats="${run}/dataset_statistics.json"

    if [ ! -f "${ckpt}" ]; then
        echo "[ERROR] ${name} checkpoint missing:"
        echo "        ${ckpt}"
        exit 1
    fi

    if [ ! -f "${stats}" ]; then
        echo "[ERROR] ${name} statistics missing:"
        echo "        ${stats}"
        exit 1
    fi

    echo "[OK] ${name}"
    echo "     run   : ${run}"
    echo "     ckpt  : ${ckpt}"
    echo "     stats : ${stats}"
}


# ==========================================================
# 8. Master log
# ==========================================================

MASTER_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MASTER_LOG="${RUN_ROOT}/object_full_pipeline_${MASTER_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1


echo
echo "=========================================================="
echo " LaWAM LIBERO-Object Full Seq-FT Pipeline"
echo "=========================================================="
echo
echo "Protocol:"
echo "  Base : Object 0-5, 10K steps"
echo "  CL1  : Object 6,   2K steps"
echo "  CL2  : Object 7,   2K steps"
echo "  CL3  : Object 8,   2K steps"
echo "  CL4  : Object 9,   2K steps"
echo
echo "Training GPUs : ${TRAIN_GPUS}"
echo
echo "Evaluation:"
echo "  Policy GPU  : ${POLICY_GPU}"
echo "  Eval GPU    : ${EVAL_GPU}"
echo "  Workers     : ${EVAL_WORKERS}"
echo "  Trials/task : ${NUM_TRIALS}"
echo "  Save videos : ${SAVE_VIDEOS}"
echo
echo "START_FROM    : ${START_FROM}"
echo "CL_START_STAGE: ${CL_START_STAGE}"
echo
echo "=========================================================="
echo


# ==========================================================
# 9. BASE
#
# LIBERO-Object:
#
# local tasks:
#   [0,1,2,3,4,5]
#
# merged indices:
#   [24,22,26,23,21,28]
#
# expected:
#   269 episodes
#   39184 frames
#
# Base-only normalization:
#   use_task_filtered_statistics=true
# ==========================================================

if [ "${START_FROM}" = "base" ]; then

    echo
    echo
    echo "=========================================================="
    echo " Starting LIBERO-Object Base Training"
    echo "=========================================================="
    echo
    echo "Tasks            : 0-5"
    echo "Steps            : 10000"
    echo "Training GPUs    : ${TRAIN_GPUS}"
    echo "Batch / GPU      : 32"
    echo "Gradient accum.  : 2"
    echo "Global batch     : 256"
    echo "Warmup           : 600"
    echo
    echo "=========================================================="
    echo


    (
        export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
        export NUM_PROCESSES=4

        bash train_lawam.sh \
            --run_root_dir="${RUN_ROOT}" \
            --run_id=base_t0_5_10k_4gpu_bs32_ga2 \
            \
            --datasets.vla_data.cl_suite=libero_object \
            "--datasets.vla_data.cl_task_ids=[0,1,2,3,4,5]" \
            \
            --datasets.vla_data.use_task_filtered_statistics=true \
            \
            --trainer.pretrained_checkpoint="${PRETRAIN_CKPT}" \
            --trainer.use_pretrained_dataset_statistics=false \
            \
            --datasets.vla_data.per_device_batch_size=32 \
            --datasets.vla_data.num_workers=4 \
            --datasets.vla_data.val_num_workers=2 \
            --datasets.vla_data.persistent_workers=true \
            \
            --trainer.gradient_accumulation_steps=2 \
            --trainer.max_train_steps=10000 \
            --trainer.num_warmup_steps=600 \
            \
            --trainer.logging_frequency=100 \
            --trainer.eval_interval=500 \
            --trainer.eval_batches=20 \
            --trainer.save_interval=2000
    )


    BASE_RUN=$(find_latest_run \
        '*+base_t0_5_10k_4gpu_bs32_ga2'
    )

    verify_run \
        "Object Base" \
        "${BASE_RUN}"


else

    BASE_RUN=$(find_latest_run \
        '*+base_t0_5_10k_4gpu_bs32_ga2'
    )

    verify_run \
        "Existing Object Base" \
        "${BASE_RUN}"

fi


# ==========================================================
# 10. CL1 -> CL4
# ==========================================================

if [ "${START_FROM}" != "eval" ]; then

    echo
    echo
    echo "=========================================================="
    echo " Starting sequential CL training"
    echo "=========================================================="
    echo
    echo "Base run:"
    echo "  ${BASE_RUN}"
    echo
    echo "Starting CL stage:"
    echo "  ${CL_START_STAGE}"
    echo
    echo "=========================================================="
    echo


    BASE_RUN="${BASE_RUN}" \
    GPUS="${TRAIN_GPUS}" \
    START_STAGE="${CL_START_STAGE}" \
    bash "${CL_SCRIPT}"

fi


# ==========================================================
# 11. Verify complete checkpoint chain
# ==========================================================

echo
echo
echo "=========================================================="
echo " Verifying complete training chain"
echo "=========================================================="
echo


BASE_RUN=$(find_latest_run \
    '*+base_t0_5_10k_4gpu_bs32_ga2'
)

CL1_RUN=$(find_latest_run \
    '*+cl1_t6_2k_4gpu_bs32_ga2'
)

CL2_RUN=$(find_latest_run \
    '*+cl2_t7_2k_4gpu_bs32_ga2'
)

CL3_RUN=$(find_latest_run \
    '*+cl3_t8_2k_4gpu_bs32_ga2'
)

CL4_RUN=$(find_latest_run \
    '*+cl4_t9_2k_4gpu_bs32_ga2'
)


verify_run "Base" "${BASE_RUN}"
verify_run "CL1"  "${CL1_RUN}"
verify_run "CL2"  "${CL2_RUN}"
verify_run "CL3"  "${CL3_RUN}"
verify_run "CL4"  "${CL4_RUN}"


echo
echo "[OK] Complete Object checkpoint chain exists."
echo


# ==========================================================
# 12. Full evaluation
#
# Base -> tasks 0-5   = 300 episodes
# CL1  -> tasks 0-6   = 350 episodes
# CL2  -> tasks 0-7   = 400 episodes
# CL3  -> tasks 0-8   = 450 episodes
# CL4  -> tasks 0-9   = 500 episodes
#
# Total = 2000 episodes
# ==========================================================

echo
echo
echo "=========================================================="
echo " Starting full LIBERO-Object CL evaluation"
echo "=========================================================="
echo
echo "Policy GPU       : ${POLICY_GPU}"
echo "Evaluator GPU    : ${EVAL_GPU}"
echo "Workers          : ${EVAL_WORKERS}"
echo "Trials / task    : ${NUM_TRIALS}"
echo "Save videos      : ${SAVE_VIDEOS}"
echo
echo "=========================================================="
echo


# Do not leak the training CUDA_VISIBLE_DEVICES mapping into
# the evaluator launcher.
unset CUDA_VISIBLE_DEVICES || true
unset NUM_PROCESSES || true


GPU_ID="${POLICY_GPU}" \
EVAL_GPU_ID="${EVAL_GPU}" \
NUM_WORKERS="${EVAL_WORKERS}" \
NUM_TRIALS="${NUM_TRIALS}" \
SAVE_VIDEOS="${SAVE_VIDEOS}" \
bash "${EVAL_SCRIPT}"


# ==========================================================
# 13. Complete
# ==========================================================

echo
echo
echo "=========================================================="
echo " LIBERO-Object full Seq-FT experiment completed"
echo "=========================================================="
echo
echo "Training:"
echo "  Base -> CL1 -> CL2 -> CL3 -> CL4"
echo
echo "Evaluation:"
echo "  Base : tasks 0-5"
echo "  CL1  : tasks 0-6"
echo "  CL2  : tasks 0-7"
echo "  CL3  : tasks 0-8"
echo "  CL4  : tasks 0-9"
echo
echo "Master log:"
echo "  ${MASTER_LOG}"
echo
echo "=========================================================="

