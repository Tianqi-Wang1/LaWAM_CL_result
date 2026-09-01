#!/usr/bin/env bash
set -euo pipefail

# Complete FORMAL Routing-V1 serial chain.
# Execution order is serial, but T6/T7/T8/T9 experts are ALL fresh from the SAME
# 10K Base (not sequential checkpoint chaining).
#
# Phase A: ALL TRAINING ONLY on the training GPU set
#   Base T0-T5 10K -> B1 T6-T9 2K each -> B2 T6-T9 2K each
# Phase B: ONLY AFTER ALL TRAINING, run all closed-loop LIBERO evaluations.
# This mirrors the smoke-chain GPU-occupancy strategy and avoids switching from
# 4-GPU training to 2-GPU evaluation in the middle of the experiment.

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

RUN_BASE="${RUN_BASE:-true}"
RUN_B1="${RUN_B1:-true}"
RUN_B2="${RUN_B2:-true}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-true}"
TASKS="${TASKS:-6 7 8 9}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"

cat <<MSG
==========================================================
 FORMAL ROUTING-V1 SERIAL CHAIN
 Phase A training only:
   Base T0-T5 : 10K
   B1 tasks    : ${TASKS}, 2K each, fresh from same Base
   B2 tasks    : ${TASKS}, 2K each, fresh from same Base
 Train GPUs    : ${TRAIN_GPUS}
 Phase B       : all closed-loop evaluation only after training
 Eval GPUs     : policy=${POLICY_GPU}, simulator=${EVAL_GPU}
 Trials/task   : ${NUM_TRIALS}
==========================================================
MSG

# -----------------------------------------------------------------------------
# Phase A: TRAIN ONLY. No LIBERO closed-loop evaluation and no B2 eager merge.
# -----------------------------------------------------------------------------
if [ "${RUN_BASE,,}" = "true" ]; then
  echo "[PHASE A1] Base T0-T5 10K training only"
  DO_EVAL=false TRAIN_GPUS="${TRAIN_GPUS}" \
    bash scripts/run_libero_goal_routing_v1_base_10k.sh
fi

if [ "${RUN_B1,,}" = "true" ]; then
  echo "[PHASE A2] B1 T6-T9 training only"
  DO_EVAL=false TASKS="${TASKS}" TRAIN_GPUS="${TRAIN_GPUS}" \
    bash scripts/run_libero_goal_routing_v1_b1_2k.sh
fi

if [ "${RUN_B2,,}" = "true" ]; then
  echo "[PHASE A3] B2 T6-T9 training only; keep LoRA checkpoints unmerged"
  DO_EVAL=false DO_MERGE=false TASKS="${TASKS}" TRAIN_GPUS="${TRAIN_GPUS}" \
    bash scripts/run_libero_goal_routing_v1_b2_2k.sh
fi

echo "=========================================================="
echo " ALL FORMAL TRAINING FINISHED. Training GPUs can now be released/shrunk."
echo "=========================================================="

# -----------------------------------------------------------------------------
# Phase B: FINAL CLOSED-LOOP EVALUATION ONLY.
# -----------------------------------------------------------------------------
if [ "${RUN_FINAL_EVAL,,}" = "true" ]; then
  TASKS="${TASKS}" \
  EVAL_BASE=true \
  EVAL_B1="${RUN_B1}" \
  EVAL_B2="${RUN_B2}" \
  NUM_TRIALS="${NUM_TRIALS}" EVAL_WORKERS="${EVAL_WORKERS}" \
  POLICY_GPU="${POLICY_GPU}" EVAL_GPU="${EVAL_GPU}" \
    bash scripts/eval_libero_goal_routing_v1_all_base10k_expert2k.sh
else
  echo "[INFO] RUN_FINAL_EVAL=false; training completed and formal checkpoints are retained."
fi

echo "=========================================================="
echo " COMPLETE FORMAL ROUTING-V1 SERIAL CHAIN"
echo "=========================================================="
