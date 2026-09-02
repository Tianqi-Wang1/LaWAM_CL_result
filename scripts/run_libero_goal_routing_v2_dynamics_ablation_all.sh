#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
TASKS="${TASKS:-6 7 8 9}"
VARIANTS="${VARIANTS:-taskwm_hdhz taskwm_hdh taskwm_dh basewm_hdhz basewm_hdh basewm_dh}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-100}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"

STAMP=$(date +"%Y%m%d_%H%M%S")
GROUP_ROOT="${V2_ROOT}/dynamics_ablation_2x3/${STAMP}"
mkdir -p "${GROUP_ROOT}"
MANIFEST="${GROUP_ROOT}/manifest.tsv"
printf 'variant\ttask\tdynamics_file\n' > "${MANIFEST}"

echo "================================================================================"
echo " Routing-V2 2x3 future-dynamics ablation — TRAIN ALL FIRST"
echo " Tasks      : ${TASKS}"
echo " Variants   : ${VARIANTS}"
echo " Steps      : ${MAX_TRAIN_STEPS} per task/NEW variant"
echo " GPU        : ${TRAIN_GPUS}"
echo " Batch      : ${PER_DEVICE_BATCH_SIZE} x 4 GPU, GA=${GRADIENT_ACCUMULATION_STEPS}"
echo " Eval loss  : every ${TRAIN_EVAL_INTERVAL} steps, ${TRAIN_EVAL_BATCHES} batches"
echo " Group root : ${GROUP_ROOT}"
echo " A1/taskwm_hdhz is existing baseline and is only extracted, not retrained."
echo "================================================================================"

for variant in ${VARIANTS}; do
  echo
  echo "################################################################################"
  echo " Variant ${variant}"
  echo "################################################################################"
  for task in ${TASKS}; do
    TASK_ID="${task}" \
    VARIANT="${variant}" \
    V2_ROOT="${V2_ROOT}" \
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
    TRAIN_GPUS="${TRAIN_GPUS}" \
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
    TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL}" \
    TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES}" \
      bash scripts/run_libero_goal_routing_v2_dynamics_variant_task.sh

    dyn="${V2_ROOT}/task${task}/routing_memory_variants/${variant}/dynamics_ae.pt"
    [ -f "${dyn}" ] || { echo "[ERROR] Missing output: ${dyn}"; exit 1; }
    printf '%s\tT%s\t%s\n' "${variant}" "${task}" "${dyn}" >> "${MANIFEST}"
  done
done

python scripts/audit_routing_v2_dynamics_ablation.py \
  --v2-root "${V2_ROOT}" --tasks ${TASKS} --variants ${VARIANTS} \
  --output "${GROUP_ROOT}/audit.csv"

ln -sfn "${GROUP_ROOT}" "${V2_ROOT}/dynamics_ablation_2x3/latest"
echo "${GROUP_ROOT}" > "${V2_ROOT}/latest_dynamics_ablation_2x3.txt"

echo "================================================================================"
echo " Routing-V2 2x3 Dynamics training COMPLETE"
echo " Manifest : ${MANIFEST}"
echo " Audit    : ${GROUP_ROOT}/audit.csv"
echo " Next     : passive 2x3 evaluation (do not launch closed-loop yet)"
echo "================================================================================"
