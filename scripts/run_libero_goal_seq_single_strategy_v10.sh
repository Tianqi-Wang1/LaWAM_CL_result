#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-b1}"
TASKS="${TASKS:-6 7 8 9}"
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
RESULT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seq_single_task_nonlinear_10k_v10"
cd "${ROOT}"

case "${MODE}" in b1|b2|b3) ;; *) echo "[ERROR] MODE must be b1/b2/b3"; exit 2 ;; esac

echo "=========================================================="
echo " ${MODE^^}: Fresh single-task sweep over seq tasks"
echo " Tasks: ${TASKS}"
echo " Every task starts from the SAME Formal Base."
echo " This is NOT sequential checkpoint chaining."
echo "=========================================================="

for task in ${TASKS}; do
  echo
  echo "#################### ${MODE^^} / T${task} ####################"
  MODE="${MODE}" TASK_ID="${task}" bash scripts/run_libero_goal_seq_single_variant_v10.sh
  echo "################## DONE ${MODE^^} / T${task} ##################"
done

python scripts/aggregate_seq_single_task_v10.py \
  --mode "${MODE}" \
  --root "${RESULT_ROOT}" \
  --tasks ${TASKS}

echo
necho=echo
${necho} "[OK] ${MODE^^} seq-task single-task sweep complete."
${necho} "Summary: ${RESULT_ROOT}/${MODE}/seq_single_task_summary.csv"
