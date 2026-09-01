#!/usr/bin/env bash
set -euo pipefail

# Complete serial Routing-V1 smoke chain.
# IMPORTANT: "serial" refers to execution order only. T6/T7/T8/T9 experts are
# ALL fresh from the SAME smoke Base, exactly matching the formal B1/B2 protocol.
# Closed-loop LIBERO evaluation is deferred until ALL training jobs have finished,
# so the 4-GPU training footprint is not interrupted by 2-GPU evaluation jobs.

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SMOKE_ROOT="${SMOKE_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_smoke_chain}"
SMOKE_BASE_STEPS="${SMOKE_BASE_STEPS:-20}"
SMOKE_EXPERT_STEPS="${SMOKE_EXPERT_STEPS:-20}"
SMOKE_BASE_WARMUP="${SMOKE_BASE_WARMUP:-2}"
SMOKE_EXPERT_WARMUP="${SMOKE_EXPERT_WARMUP:-2}"
TASKS="${TASKS:-6 7 8 9}"
SMOKE_MODES="${SMOKE_MODES:-b1 b2}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SMOKE_NUM_TRIALS="${SMOKE_NUM_TRIALS:-1}"
SMOKE_EVAL_WORKERS="${SMOKE_EVAL_WORKERS:-4}"
DO_FINAL_EVAL="${DO_FINAL_EVAL:-true}"
RESET_SMOKE_ROOT="${RESET_SMOKE_ROOT:-true}"
CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL="${CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL:-true}"

# Smoke runs are disposable; clear a previous smoke chain by default to avoid stale
# run discovery and unnecessary disk use. Safety guard prevents deleting a non-smoke path.
if [ "${RESET_SMOKE_ROOT,,}" = "true" ] && [ -e "${SMOKE_ROOT}" ]; then
  case "${SMOKE_ROOT}" in
    *routing_v1_smoke*)
      echo "[CLEAN] Removing previous smoke root: ${SMOKE_ROOT}"
      rm -rf "${SMOKE_ROOT}"
      ;;
    *)
      echo "[ERROR] Refusing RESET_SMOKE_ROOT for non-smoke-looking path: ${SMOKE_ROOT}"
      exit 1
      ;;
  esac
fi
mkdir -p "${SMOKE_ROOT}"

cat <<MSG
==========================================================
 COMPLETE ROUTING-V1 SERIAL SMOKE CHAIN
 Smoke Base       : T0-T5, ${SMOKE_BASE_STEPS} steps
 Expert tasks     : ${TASKS}
 Expert modes     : ${SMOKE_MODES}
 Expert steps     : ${SMOKE_EXPERT_STEPS} each
 Train GPUs       : ${TRAIN_GPUS}
 Evaluation       : ONLY AFTER all training finishes
 Final trials     : ${SMOKE_NUM_TRIALS}/task
 Smoke root       : ${SMOKE_ROOT}
 Auto-clean ckpts : ${CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL}
==========================================================
MSG

# -----------------------------------------------------------------------------
# Phase A: TRAIN ONLY. Keep the same four-GPU footprint throughout the chain.
# -----------------------------------------------------------------------------
echo "[PHASE A1] Smoke Base training (NO closed-loop eval, NO train-time val)"
SMOKE_ROOT="${SMOKE_ROOT}" \
SMOKE_STEPS="${SMOKE_BASE_STEPS}" \
SMOKE_WARMUP_STEPS="${SMOKE_BASE_WARMUP}" \
SMOKE_TRAIN_EVAL_INTERVAL="$((SMOKE_BASE_STEPS + 1))" \
SMOKE_TRAIN_EVAL_BATCHES=1 \
SMOKE_DO_EVAL=false \
TRAIN_GPUS="${TRAIN_GPUS}" \
bash scripts/run_libero_goal_routing_v1_base_smoke.sh

BASE_RUN=$(cat "${SMOKE_ROOT}/latest_base_run.txt")
[ -f "${BASE_RUN}/final_model/pytorch_model.pt" ] || { echo "[ERROR] Smoke Base checkpoint missing"; exit 1; }
echo "[OK] Smoke Base: ${BASE_RUN}"

for mode in ${SMOKE_MODES}; do
  echo "[PHASE A2] ${mode^^} smoke experts T6-T9 (fresh from same Base; NO eval)"
  for task in ${TASKS}; do
    SMOKE_ROOT="${SMOKE_ROOT}" \
    BASE_RUN="${BASE_RUN}" \
    MODE="${mode}" TASK_ID="${task}" \
    SMOKE_EXPERT_STEPS="${SMOKE_EXPERT_STEPS}" \
    SMOKE_EXPERT_WARMUP="${SMOKE_EXPERT_WARMUP}" \
    TRAIN_GPUS="${TRAIN_GPUS}" \
    bash scripts/run_libero_goal_routing_v1_expert_smoke.sh
  done
done

# Training is now fully finished. Only at this point do we shrink to the eval GPU pair.
echo "=========================================================="
echo " ALL SMOKE TRAINING FINISHED. Starting FINAL evaluation phase."
echo "=========================================================="

# -----------------------------------------------------------------------------
# Phase B: EVAL ONLY.
# -----------------------------------------------------------------------------
if [ "${DO_FINAL_EVAL,,}" = "true" ]; then
  SMOKE_ROOT="${SMOKE_ROOT}" \
  BASE_RUN="${BASE_RUN}" \
  TASKS="${TASKS}" \
  SMOKE_MODES="${SMOKE_MODES}" \
  SMOKE_NUM_TRIALS="${SMOKE_NUM_TRIALS}" \
  SMOKE_EVAL_WORKERS="${SMOKE_EVAL_WORKERS}" \
  POLICY_GPU="${POLICY_GPU}" EVAL_GPU="${EVAL_GPU}" \
  CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL="${CLEAN_SMOKE_CHECKPOINTS_AFTER_EVAL}" \
  bash scripts/eval_libero_goal_routing_v1_smoke_chain.sh
else
  echo "[INFO] DO_FINAL_EVAL=false; all smoke checkpoints remain under ${SMOKE_ROOT}."
fi

echo "=========================================================="
echo " COMPLETE SERIAL SMOKE CHAIN PASSED"
echo " Next formal command:"
echo "   bash scripts/run_libero_goal_routing_v1_base_10k.sh"
echo "=========================================================="
