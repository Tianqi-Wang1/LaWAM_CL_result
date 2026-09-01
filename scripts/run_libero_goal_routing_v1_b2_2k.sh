#!/usr/bin/env bash
set -euo pipefail
TASKS="${TASKS:-6 7 8 9}"
for task in ${TASKS}; do
  MODE=b2 TASK_ID="${task}" bash scripts/run_libero_goal_routing_v1_seq_single_variant_2k.sh
done
