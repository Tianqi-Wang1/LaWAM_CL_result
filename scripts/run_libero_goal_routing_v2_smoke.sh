#!/usr/bin/env bash
set -euo pipefail
# Full wiring smoke: train EVERYTHING first, evaluate only at the end.
BASE_STEPS="${BASE_STEPS:-20}" SKILL_STEPS="${SKILL_STEPS:-20}" MEMORY_STEPS="${MEMORY_STEPS:-20}" NUM_TRIALS="${NUM_TRIALS:-1}" \
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}" EVAL_WORKERS="${EVAL_WORKERS:-4}" \
bash scripts/run_libero_goal_routing_v2_all.sh
