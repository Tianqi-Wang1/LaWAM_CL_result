#!/usr/bin/env bash
set -euo pipefail

# FINAL-ONLY closed-loop evaluation for the formal Routing-V1 chain.
# Expected training protocol before this script:
#   Base T0-T5: 10K
#   B1 T6-T9: fresh from the same Base, 2K each
#   B2 T6-T9: fresh from the same Base, 2K each
# All training is finished before this script starts any LIBERO evaluation.
# B2 LoRA checkpoints are merged just-in-time, evaluated, then the temporary
# merged checkpoint is deleted to save disk. The original unmerged final remains.

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
ROUTING_ROOT="${ROUTING_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_base10k_expert2k}"
TASKS="${TASKS:-6 7 8 9}"
EVAL_BASE="${EVAL_BASE:-true}"
EVAL_B1="${EVAL_B1:-true}"
EVAL_B2="${EVAL_B2:-true}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DELETE_TEMP_MERGED="${DELETE_TEMP_MERGED:-true}"
VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
EVAL_STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
EVAL_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_base10k_expert2k/final_only/${EVAL_STAMP}"
mkdir -p "${EVAL_ROOT}"

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

BASE_RUN="${BASE_RUN:-}"
if [ -z "${BASE_RUN}" ] && [ -f "${ROUTING_ROOT}/latest_base_run.txt" ]; then
  BASE_RUN=$(cat "${ROUTING_ROOT}/latest_base_run.txt")
fi
if [ -z "${BASE_RUN}" ]; then
  BASE_RUN=$(find "${ROUTING_ROOT}/base_runs" -maxdepth 1 -type d -name '*+routing_v1_base_t0_5_*' 2>/dev/null | sort | tail -n 1)
fi
[ -n "${BASE_RUN}" ] || { echo "[ERROR] Formal Routing-V1 Base run not found"; exit 1; }
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
[ -f "${BASE_CKPT}" ] || { echo "[ERROR] Missing Base checkpoint: ${BASE_CKPT}"; exit 1; }

find_latest_run() {
  local mode="$1" task="$2"
  find "${ROUTING_ROOT}/${mode}/task${task}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

eval_one() {
  local alias="$1" task_ids="$2" ckpt="$3"
  local out="${EVAL_ROOT}/${alias}"
  mkdir -p "${out}"
  echo "=========================================================="
  echo " FINAL FORMAL EVAL: ${alias}"
  echo " Tasks      : ${task_ids}"
  echo " Checkpoint : ${ckpt}"
  echo " Trials/task: ${NUM_TRIALS}"
  echo " Eval GPUs  : policy=${POLICY_GPU}, simulator=${EVAL_GPU}"
  echo "=========================================================="
  SUITES="libero_goal" TASK_IDS="${task_ids}" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
  NUM_WORKERS="${EVAL_WORKERS}" GPU_IDS="${POLICY_GPU}" EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" OUTPUT_ROOT="${out}" LIBERO_CKPT_ALIAS="${alias}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${ckpt}"

  local eval_dir suite_dir
  eval_dir=$(find "${out}/${alias}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  suite_dir="${eval_dir}/suites/libero_goal"
  python scripts/summarize_libero_cl_eval.py \
    --run-dir "${suite_dir}" --task-ids ${task_ids} --expected-trials "${NUM_TRIALS}"
  echo "${alias},${task_ids},${ckpt},${suite_dir}/per_task_summary.csv" >> "${EVAL_ROOT}/manifest.csv"
}

echo "alias,task_ids,checkpoint,summary_csv" > "${EVAL_ROOT}/manifest.csv"

# Base is evaluated only now, after the whole requested training phase is complete.
if [ "${EVAL_BASE,,}" = "true" ]; then
  eval_one "routing_v1_base_t0_5" "0 1 2 3 4 5" "${BASE_CKPT}"
fi

if [ "${EVAL_B1,,}" = "true" ]; then
  for task in ${TASKS}; do
    run=$(find_latest_run b1 "${task}")
    [ -n "${run}" ] || { echo "[ERROR] Missing B1 T${task} run"; exit 1; }
    raw="${run}/final_model/pytorch_model.pt"
    [ -f "${raw}" ] || { echo "[ERROR] Missing ${raw}"; exit 1; }
    eval_one "routing_v1_b1_t${task}" "${task}" "${raw}"
  done
fi

if [ "${EVAL_B2,,}" = "true" ]; then
  for task in ${TASKS}; do
    run=$(find_latest_run b2 "${task}")
    [ -n "${run}" ] || { echo "[ERROR] Missing B2 T${task} run"; exit 1; }
    raw="${run}/final_model/pytorch_model.pt"
    [ -f "${raw}" ] || { echo "[ERROR] Missing ${raw}"; exit 1; }
    tmp_merged="${run}/final_model/pytorch_model_eval_merged.pt"
    rm -f "${tmp_merged}"
    python scripts/merge_lora_checkpoint_v6.py \
      --input "${raw}" --output "${tmp_merged}" --alpha "${VLM_LORA_ALPHA}"
    eval_one "routing_v1_b2_t${task}" "${task}" "${tmp_merged}"
    if [ "${DELETE_TEMP_MERGED,,}" = "true" ]; then
      rm -f "${tmp_merged}"
      echo "[CLEAN] Removed temporary merged B2 checkpoint: ${tmp_merged}"
    fi
  done
fi

echo "=========================================================="
echo " FINAL FORMAL EVALUATION COMPLETE"
echo " Manifest : ${EVAL_ROOT}/manifest.csv"
echo " Eval root: ${EVAL_ROOT}"
echo "=========================================================="
