#!/usr/bin/env bash
set -euo pipefail
# Wall-clock serial chain. IMPORTANT: T6-T9 are independent skill paths; each is
# initialized fresh from the SAME V2 Base, not from the previous task checkpoint.
BASE_STEPS="${BASE_STEPS:-10000}"; SKILL_STEPS="${SKILL_STEPS:-2000}"; MEMORY_STEPS="${MEMORY_STEPS:-1000}"; NUM_TRIALS="${NUM_TRIALS:-50}"
SKIP_BASE="${SKIP_BASE:-false}"
if [ "${SKIP_BASE}" = "true" ]; then
  V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
  [ -f "${V2_ROOT}/latest_base_run.txt" ] || { echo "[ERROR] SKIP_BASE=true but ${V2_ROOT}/latest_base_run.txt is missing"; exit 1; }
  BASE_RUN="$(cat "${V2_ROOT}/latest_base_run.txt")"
  [ -f "${BASE_RUN}/final_model/pytorch_model.pt" ] || { echo "[ERROR] Existing V2 Base checkpoint missing: ${BASE_RUN}/final_model/pytorch_model.pt"; exit 1; }
  echo "[RoutingV2] SKIP_BASE=true; reusing existing Base: ${BASE_RUN}"
else
  MAX_TRAIN_STEPS="${BASE_STEPS}" bash scripts/run_libero_goal_routing_v2_base.sh
fi
for t in 6 7 8 9; do
  TASK_ID="${t}" MAX_TRAIN_STEPS="${SKILL_STEPS}" bash scripts/run_libero_goal_routing_v2_skill_task.sh
  TASK_ID="${t}" MAX_TRAIN_STEPS="${MEMORY_STEPS}" bash scripts/run_libero_goal_routing_v2_memory_task.sh
done
NUM_TRIALS="${NUM_TRIALS}" bash scripts/eval_libero_goal_routing_v2_taskid_all.sh
