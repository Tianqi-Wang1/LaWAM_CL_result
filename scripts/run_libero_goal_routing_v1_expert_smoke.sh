#!/usr/bin/env bash
set -euo pipefail

# Smoke one Routing-V1 expert without closed-loop evaluation.
# The task is still FRESH from the same latent-enabled smoke Base; this is NOT checkpoint chaining.
# All closed-loop evaluations are intentionally deferred to the final smoke-chain evaluation phase.

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
VARIANT_SCRIPT="${ROOT}/scripts/run_libero_goal_routing_v1_seq_single_variant_2k.sh"
[ -f "${VARIANT_SCRIPT}" ] || { echo "[ERROR] Missing ${VARIANT_SCRIPT}"; exit 1; }

MODE="${MODE:-b1}"
TASK_ID="${TASK_ID:-9}"
SMOKE_ROOT="${SMOKE_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_smoke_chain}"
SMOKE_EXPERT_STEPS="${SMOKE_EXPERT_STEPS:-20}"
SMOKE_EXPERT_WARMUP="${SMOKE_EXPERT_WARMUP:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

case "${MODE}" in b1|b2) ;; *) echo "[ERROR] MODE must be b1 or b2"; exit 2 ;; esac
case "${TASK_ID}" in 6|7|8|9) ;; *) echo "[ERROR] TASK_ID must be 6/7/8/9"; exit 2 ;; esac

if [ -z "${BASE_RUN:-}" ] && [ -f "${SMOKE_ROOT}/latest_base_run.txt" ]; then
  BASE_RUN=$(cat "${SMOKE_ROOT}/latest_base_run.txt")
fi
[ -n "${BASE_RUN:-}" ] || { echo "[ERROR] Smoke Base not found under ${SMOKE_ROOT}"; exit 1; }
[ -f "${BASE_RUN}/final_model/pytorch_model.pt" ] || { echo "[ERROR] Invalid BASE_RUN=${BASE_RUN}"; exit 1; }

cat <<MSG
==========================================================
 Routing-V1 ${MODE^^} EXPERT SMOKE / T${TASK_ID}
 Fresh Base       : ${BASE_RUN}
 Optimizer steps  : ${SMOKE_EXPERT_STEPS}
 Train GPUs       : ${TRAIN_GPUS}
 Per-device BS    : ${PER_DEVICE_BATCH_SIZE}
 Grad accum       : ${GRADIENT_ACCUMULATION_STEPS}
 Train-time val   : disabled for smoke chain
 Closed-loop eval : deferred to FINAL phase
 B2 merge         : deferred to FINAL phase
==========================================================
MSG

ROUTING_ROOT="${SMOKE_ROOT}" \
BASE_RUN="${BASE_RUN}" \
MODE="${MODE}" \
TASK_ID="${TASK_ID}" \
TRAIN_GPUS="${TRAIN_GPUS}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
MAX_TRAIN_STEPS="${SMOKE_EXPERT_STEPS}" \
NUM_WARMUP_STEPS="${SMOKE_EXPERT_WARMUP}" \
TRAIN_EVAL_INTERVAL="$((SMOKE_EXPERT_STEPS + 1))" \
TRAIN_EVAL_BATCHES=1 \
LOGGING_FREQUENCY=1 \
SAVE_INTERVAL="$((SMOKE_EXPERT_STEPS + 1))" \
DO_EVAL=false \
DO_MERGE=false \
bash "${VARIANT_SCRIPT}"
