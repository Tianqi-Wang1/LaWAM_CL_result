#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# R-Base: LIBERO-Goal T9
#         Base upstream + FROZEN Base Flow + task-specific Residual Expert
#         Default: last 4 DiT blocks receive nonlinear residual branches (R4).
#
# Parallel-friendly: auto-preserve global batch=256, dedicated DDP port 29631,
# DO_EVAL=false for train-only, DO_TRAIN=false for eval-only.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
VERIFY_SCRIPT="${ROOT}/scripts/verify_flow_residual_experiment.py"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_residual_expert_split_v1/residual_baseflow"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_residual_expert_split_v1/residual_baseflow"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
[ "${NUM_TRAIN_GPUS}" -gt 0 ] || { echo "[ERROR] TRAIN_GPUS is empty"; exit 1; }

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-256}"
if [ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]; then
    DENOM=$((PER_DEVICE_BATCH_SIZE * NUM_TRAIN_GPUS))
    if [ $((TARGET_GLOBAL_BATCH_SIZE % DENOM)) -ne 0 ]; then
        echo "[ERROR] TARGET_GLOBAL_BATCH_SIZE=${TARGET_GLOBAL_BATCH_SIZE} is not divisible by PER_DEVICE_BATCH_SIZE*NUM_GPUS=${DENOM}."
        exit 1
    fi
    GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / DENOM))
fi
GLOBAL_BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_GPUS))

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
ACTION_LR="${ACTION_LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-4}"
DO_TRAIN="${DO_TRAIN:-true}"
DO_EVAL="${DO_EVAL:-true}"
RUN_DIR_OVERRIDE="${RUN_DIR_OVERRIDE:-}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29631}"

[[ "${RESIDUAL_BLOCKS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] RESIDUAL_BLOCKS must be a positive integer"; exit 1; }

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

find_run() {
    local root="$1" pattern="$2"
    find "${root}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}
verify_run() {
    local label="$1" run="$2"
    [ -n "${run}" ] || { echo "[ERROR] ${label}: run not found"; exit 1; }
    for f in "${run}/config.yaml" "${run}/dataset_statistics.json" "${run}/final_model/pytorch_model.pt"; do
        [ -f "${f}" ] || { echo "[ERROR] ${label}: missing ${f}"; exit 1; }
    done
    echo "[OK] ${label}: ${run}"
}
verify_stats() {
    local reference="$1" current="$2"
    python - "${reference}" "${current}" <<'PY'
import json,sys
rp,cp=sys.argv[1:3]
with open(rp,"r",encoding="utf-8") as f:r=json.load(f)
with open(cp,"r",encoding="utf-8") as f:c=json.load(f)
for tag in r:
    for sec in ("action","state"):
        if r[tag][sec] != c[tag][sec]:
            raise RuntimeError(f"Normalization statistics changed: {tag}/{sec}")
print("[OK] action/state normalization is identical to Base.")
PY
}

if [ -z "${BASE_RUN:-}" ]; then
    BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Formal Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"
[ -f "${SUMMARY_SCRIPT}" ] || { echo "[ERROR] Missing ${SUMMARY_SCRIPT}"; exit 1; }
[ -f "${VERIFY_SCRIPT}" ] || { echo "[ERROR] Missing ${VERIFY_SCRIPT}"; exit 1; }

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/residual_r${RESIDUAL_BLOCKS}_baseflow_${PIPELINE_TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1
RUN_ID="t9_residual_r${RESIDUAL_BLOCKS}_baseflow_${MAX_TRAIN_STEPS}step_gbs${GLOBAL_BATCH_SIZE}"

echo "=========================================================="
echo " R-Base: Frozen Base Flow + R${RESIDUAL_BLOCKS} -> T9"
echo "=========================================================="
echo "Base checkpoint : ${BASE_CKPT}"
echo "Frozen Flow     : BASE Flow"
echo "Residual blocks : ${RESIDUAL_BLOCKS}"
echo "Train GPUs      : ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "Per-device BS   : ${PER_DEVICE_BATCH_SIZE}"
echo "Grad accum      : ${GRADIENT_ACCUMULATION_STEPS}"
echo "Global batch    : ${GLOBAL_BATCH_SIZE}"
echo "Steps / warmup  : ${MAX_TRAIN_STEPS} / ${NUM_WARMUP_STEPS}"
echo "Action LR       : ${ACTION_LR}"
echo "DDP port        : ${MAIN_PROCESS_PORT}"
echo "DO_TRAIN/EVAL   : ${DO_TRAIN}/${DO_EVAL}"
echo "=========================================================="

RUN=""
if [ "${DO_TRAIN}" = "true" ]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES="${NUM_TRAIN_GPUS}"

    bash train_lawam.sh \
        "--run_root_dir=${RUN_ROOT}" \
        "--run_id=${RUN_ID}" \
        "--datasets.vla_data.cl_suite=libero_goal" \
        "--datasets.vla_data.cl_task_ids=[9]" \
        "--datasets.vla_data.use_task_filtered_statistics=false" \
        "--trainer.use_pretrained_dataset_statistics=true" \
        "--trainer.pretrained_checkpoint=${BASE_CKPT}" \
        "--trainer.load_pretrained_policy_flow=true" \
        "--trainer.policy_flow_override_checkpoint=null" \
        "--framework.action_model.flow_cfg.residual_expert_num_blocks=${RESIDUAL_BLOCKS}" \
        "--framework.action_model.flow_cfg.residual_expert_zero_init=true" \
        "--framework.action_model.flow_cfg.residual_expert_scale=1.0" \
        "--trainer.freeze.unfreeze_lam_decoder=false" \
        "--trainer.freeze.train_flow_only=false" \
        "--trainer.freeze.train_flow_lora=false" \
        "--trainer.freeze.train_flow_residual_expert=true" \
        "--trainer.learning_rate.vlm.lr=${ACTION_LR}" \
        "--trainer.learning_rate.action_model.lr=${ACTION_LR}" \
        "--trainer.learning_rate.world_model.lr=${ACTION_LR}" \
        "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
        "--datasets.vla_data.num_workers=${NUM_WORKERS}" \
        "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}" \
        "--datasets.vla_data.persistent_workers=true" \
        "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
        "--trainer.max_train_steps=${MAX_TRAIN_STEPS}" \
        "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}" \
        "--trainer.logging_frequency=${LOGGING_FREQUENCY}" \
        "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}" \
        "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}" \
        "--trainer.save_interval=${SAVE_INTERVAL}"

    RUN=$(find_run "${RUN_ROOT}" "*+${RUN_ID}")
else
    if [ -n "${RUN_DIR_OVERRIDE}" ]; then
        RUN="${RUN_DIR_OVERRIDE}"
    else
        RUN=$(find_run "${RUN_ROOT}" "*+${RUN_ID}")
    fi
fi

verify_run "Residual R${RESIDUAL_BLOCKS} Base-Flow" "${RUN}"
verify_stats "${BASE_STATS}" "${RUN}/dataset_statistics.json"
python "${VERIFY_SCRIPT}" \
    --mode residual \
    --base-ckpt "${BASE_CKPT}" \
    --flow-source-ckpt "${BASE_CKPT}" \
    --final-ckpt "${RUN}/final_model/pytorch_model.pt" \
    --num-residual-blocks "${RESIDUAL_BLOCKS}"

unset CUDA_VISIBLE_DEVICES || true
unset NUM_PROCESSES || true

if [ "${DO_EVAL}" = "true" ]; then
    export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
    export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
    export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

    ALIAS="residual_r${RESIDUAL_BLOCKS}_baseflow"
    EVAL_MASTER="${OUTPUT_ROOT}/${PIPELINE_TIMESTAMP}"
    mkdir -p "${EVAL_MASTER}"
    SUITES="libero_goal" \
    TASK_IDS="9" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${EVAL_WORKERS}" \
    GPU_IDS="${POLICY_GPU}" \
    EVAL_GPU_IDS="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${EVAL_MASTER}" \
    LIBERO_CKPT_ALIAS="${ALIAS}" \
    bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${RUN}/final_model/pytorch_model.pt"

    EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    [ -n "${EVAL_DIR}" ] || { echo "[ERROR] Evaluation output missing"; exit 1; }
    SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
    python "${SUMMARY_SCRIPT}" --run-dir "${SUITE_DIR}" --task-ids 9 --expected-trials "${NUM_TRIALS}"
    echo "[RESULT] ${SUITE_DIR}/per_task_summary.csv"
fi

echo "=========================================================="
echo " R-Base complete"
echo " Run       : ${RUN}"
echo " Master log: ${MASTER_LOG}"
echo "=========================================================="
