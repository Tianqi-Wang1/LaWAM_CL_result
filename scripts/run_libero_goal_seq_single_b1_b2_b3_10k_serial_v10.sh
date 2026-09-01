#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

MODES="${MODES:-b1 b2 b3}"
TASKS="${TASKS:-6 7 8 9}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-600}"
export TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-1000}"
export TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
export NUM_TRIALS="${NUM_TRIALS:-50}"

if [ "${MAX_TRAIN_STEPS}" != "10000" ]; then
  echo "[WARN] MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}; requested formal protocol is 10000."
fi

echo "=========================================================="
echo " B1/B2/B3 fresh single-task sweep @ ${MAX_TRAIN_STEPS} steps"
echo " Modes: ${MODES}"
echo " Tasks: ${TASKS}"
echo " Every task starts independently from the SAME Formal Base."
echo " Order: each strategy runs T6->T7->T8->T9, then next strategy."
echo "=========================================================="

for mode in ${MODES}; do
  echo
  echo "##########################################################"
  echo " START ${mode^^} @ ${MAX_TRAIN_STEPS} steps"
  echo "##########################################################"
  MODE="${mode}" TASKS="${TASKS}" bash scripts/run_libero_goal_seq_single_strategy_v10.sh
  echo "##########################################################"
  echo " DONE  ${mode^^} @ ${MAX_TRAIN_STEPS} steps"
  echo "##########################################################"
done

python scripts/aggregate_b1_b2_b3_10k_v10.py

echo
 echo "[OK] B1/B2/B3 10k serial sweep complete."
