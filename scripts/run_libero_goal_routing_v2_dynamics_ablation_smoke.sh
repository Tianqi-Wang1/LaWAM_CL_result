#!/usr/bin/env bash
set -euo pipefail
# Minimal wiring smoke: one non-baseline Task-WM variant and the proposed Shared-WM variant.
# This validates both new code paths without spending time on all 20 training jobs.
TASK_ID="${TASK_ID:-6}" VARIANT=taskwm_hdh MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}" \
  TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-10}" \
  bash scripts/run_libero_goal_routing_v2_dynamics_variant_task.sh
TASK_ID="${TASK_ID:-6}" VARIANT=basewm_hdh MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}" \
  TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-10}" \
  bash scripts/run_libero_goal_routing_v2_dynamics_variant_task.sh

echo "[OK] 2x3 Dynamics smoke complete for T${TASK_ID}."
