#!/usr/bin/env bash
set -euo pipefail

# Final-only evaluation phase for the complete Routing-V1 smoke chain.
# No evaluation is run during Base/B1/B2 training. B2 LoRA checkpoints are merged
# just-in-time here, evaluated, and the temporary merged checkpoint is removed.

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
SMOKE_ROOT="${SMOKE_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_smoke_chain}"
TASKS="${TASKS:-6 7 8 9}"
SMOKE_MODES="${SMOKE_MODES:-b1 b2}"
SMOKE_NUM_TRIALS="${SMOKE_NUM_TRIALS:-1}"
SMOKE_EVAL_WORKERS="${SMOKE_EVAL_WORKERS:-4}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DELETE_TEMP_MERGED="${DELETE_TEMP_MERGED:-true}"
CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL="${CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL:-true}"
EVAL_STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
EVAL_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_smoke_chain/${EVAL_STAMP}"
mkdir -p "${EVAL_ROOT}"

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

BASE_RUN="${BASE_RUN:-}"
if [ -z "${BASE_RUN}" ] && [ -f "${SMOKE_ROOT}/latest_base_run.txt" ]; then
  BASE_RUN=$(cat "${SMOKE_ROOT}/latest_base_run.txt")
fi
[ -n "${BASE_RUN}" ] || { echo "[ERROR] Missing smoke Base run"; exit 1; }
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
[ -f "${BASE_CKPT}" ] || { echo "[ERROR] Missing Base checkpoint: ${BASE_CKPT}"; exit 1; }

find_latest_run() {
  local mode="$1" task="$2"
  find "${SMOKE_ROOT}/${mode}/task${task}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

eval_one() {
  local alias="$1" task_ids="$2" ckpt="$3"
  local out="${EVAL_ROOT}/${alias}"
  mkdir -p "${out}"
  echo "=========================================================="
  echo " FINAL SMOKE EVAL: ${alias}"
  echo " Tasks      : ${task_ids}"
  echo " Checkpoint : ${ckpt}"
  echo " Trials/task: ${SMOKE_NUM_TRIALS}"
  echo "=========================================================="
  SUITES="libero_goal" TASK_IDS="${task_ids}" NUM_TRIALS_PER_TASK="${SMOKE_NUM_TRIALS}" \
  NUM_WORKERS="${SMOKE_EVAL_WORKERS}" GPU_IDS="${POLICY_GPU}" EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" OUTPUT_ROOT="${out}" LIBERO_CKPT_ALIAS="${alias}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${ckpt}"
  local eval_dir suite_dir
  eval_dir=$(find "${out}/${alias}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  suite_dir="${eval_dir}/suites/libero_goal"
  python scripts/summarize_libero_cl_eval.py \
    --run-dir "${suite_dir}" --task-ids ${task_ids} --expected-trials "${SMOKE_NUM_TRIALS}"
  echo "${alias},${task_ids},${ckpt},${suite_dir}/per_task_summary.csv" >> "${EVAL_ROOT}/manifest.csv"
}

echo "alias,task_ids,checkpoint,summary_csv" > "${EVAL_ROOT}/manifest.csv"

# 1) Base evaluation is also deferred until now.
eval_one "smoke_base_t0_5" "0 1 2 3 4 5" "${BASE_CKPT}"

# 2) Expert evaluations. B2 is merged only when its turn arrives.
for mode in ${SMOKE_MODES}; do
  case "${mode}" in b1|b2) ;; *) echo "[ERROR] Invalid smoke mode ${mode}"; exit 2 ;; esac
  for task in ${TASKS}; do
    run=$(find_latest_run "${mode}" "${task}")
    [ -n "${run}" ] || { echo "[ERROR] Missing ${mode} T${task} smoke run"; exit 1; }
    raw="${run}/final_model/pytorch_model.pt"
    [ -f "${raw}" ] || { echo "[ERROR] Missing ${raw}"; exit 1; }
    ckpt="${raw}"
    tmp_merged=""
    if [ "${mode}" = "b2" ]; then
      tmp_merged="${run}/final_model/pytorch_model_smoke_merged.pt"
      rm -f "${tmp_merged}"
      python scripts/merge_lora_checkpoint_v6.py --input "${raw}" --output "${tmp_merged}" --alpha 32
      ckpt="${tmp_merged}"
    fi
    eval_one "smoke_${mode}_t${task}" "${task}" "${ckpt}"
    if [ -n "${tmp_merged}" ] && [ "${DELETE_TEMP_MERGED,,}" = "true" ]; then
      rm -f "${tmp_merged}"
      echo "[CLEAN] Removed temporary merged B2 checkpoint: ${tmp_merged}"
    fi
  done
done

# Smoke checkpoints are disposable. Only delete them after EVERY requested eval succeeds.
if [ "${CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL,,}" = "true" ]; then
  case "${SMOKE_ROOT}" in
    *routing_v1_smoke*)
      echo "[CLEAN] All smoke evaluations passed. Removing smoke .pt checkpoints to reclaim disk."
      find "${SMOKE_ROOT}" -type f \( -name 'pytorch_model.pt' -o -name 'pytorch_model_merged.pt' -o -name 'pytorch_model_smoke_merged.pt' \) -print -delete
      ;;
    *)
      echo "[ERROR] Refusing checkpoint cleanup because SMOKE_ROOT does not look like a smoke directory: ${SMOKE_ROOT}"
      exit 1
      ;;
  esac
fi

echo "=========================================================="
echo " FINAL SMOKE EVALUATION PASSED"
echo " Manifest : ${EVAL_ROOT}/manifest.csv"
echo " Eval root: ${EVAL_ROOT}"
echo "=========================================================="
