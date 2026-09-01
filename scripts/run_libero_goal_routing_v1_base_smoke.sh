#!/usr/bin/env bash
set -euo pipefail

# Routing-V1 Base smoke test.
# Purpose: exercise the SAME Base training/freezing protocol as the formal 10K run,
# but for only a few optimizer steps, then optionally run a tiny closed-loop eval.
# This writes to a separate smoke directory and will not overwrite the formal Base run.

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
FORMAL_SCRIPT="${ROOT}/scripts/run_libero_goal_routing_v1_base_10k.sh"

[ -f "${FORMAL_SCRIPT}" ] || {
  echo "[ERROR] Missing formal Base script: ${FORMAL_SCRIPT}"
  exit 1
}

SMOKE_STEPS="${SMOKE_STEPS:-20}"
SMOKE_WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-2}"
SMOKE_TRAIN_EVAL_INTERVAL="${SMOKE_TRAIN_EVAL_INTERVAL:-10}"
SMOKE_TRAIN_EVAL_BATCHES="${SMOKE_TRAIN_EVAL_BATCHES:-2}"
SMOKE_NUM_TRIALS="${SMOKE_NUM_TRIALS:-1}"
SMOKE_EVAL_WORKERS="${SMOKE_EVAL_WORKERS:-4}"
SMOKE_DO_EVAL="${SMOKE_DO_EVAL:-true}"

# Keep the formal batch/GPU setup by default so the smoke test also checks memory
# and distributed execution under the same configuration as the 10K experiment.
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

SMOKE_ROOT="${SMOKE_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_smoke}"

cat <<MSG
==========================================================
 Routing-V1 BASE SMOKE
 Formal protocol : same as Base-10K
 Optimizer steps : ${SMOKE_STEPS}
 Warmup steps    : ${SMOKE_WARMUP_STEPS}
 Train GPUs      : ${TRAIN_GPUS}
 Per-device BS   : ${PER_DEVICE_BATCH_SIZE}
 Grad accum      : ${GRADIENT_ACCUMULATION_STEPS}
 Tiny eval       : ${SMOKE_DO_EVAL} (${SMOKE_NUM_TRIALS} trial/task)
 Output root     : ${SMOKE_ROOT}
==========================================================
MSG

# The formal script already performs:
#   1) training,
#   2) final_model existence checks,
#   3) Routing-V1 checkpoint protocol verification,
#   4) optional closed-loop LIBERO evaluation.
# We only override runtime length / logging / output location here.
ROUTING_ROOT="${SMOKE_ROOT}" \
TRAIN_GPUS="${TRAIN_GPUS}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
MAX_TRAIN_STEPS="${SMOKE_STEPS}" \
NUM_WARMUP_STEPS="${SMOKE_WARMUP_STEPS}" \
TRAIN_EVAL_INTERVAL="${SMOKE_TRAIN_EVAL_INTERVAL}" \
TRAIN_EVAL_BATCHES="${SMOKE_TRAIN_EVAL_BATCHES}" \
LOGGING_FREQUENCY=1 \
SAVE_INTERVAL="$((SMOKE_STEPS + 1))" \
DO_EVAL="${SMOKE_DO_EVAL}" \
NUM_TRIALS="${SMOKE_NUM_TRIALS}" \
EVAL_WORKERS="${SMOKE_EVAL_WORKERS}" \
bash "${FORMAL_SCRIPT}"

cat <<'MSG'
==========================================================
 SMOKE PASSED
 If the command reached this point, training completed,
 final_model was written, the Routing-V1 checkpoint verifier
 passed, and (when enabled) the tiny LIBERO eval also finished.

 You can now launch the formal Base run with:
   bash scripts/run_libero_goal_routing_v1_base_10k.sh
==========================================================
MSG
