#!/usr/bin/env bash
set -euo pipefail

# Fast diagnostic sequence recommended before any new 50-trial routing run.
# 1) 5-trial T6 single-candidate sanity for B1/B2.
# 2) 1-trial/task final-bank oracle-state score probe for B1/B2.
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

echo "==================== [1/2] single-candidate sanity ===================="
NUM_TRIALS="${SANITY_TRIALS:-5}" EVAL_WORKERS="${EVAL_WORKERS:-4}" \
  bash scripts/eval_libero_goal_routing_v1_single_candidate_sanity.sh

echo "==================== [2/2] oracle-state score probe =================="
NUM_TRIALS="${PROBE_TRIALS:-1}" EVAL_WORKERS="${EVAL_WORKERS:-4}" \
  bash scripts/eval_libero_goal_routing_v1_oracle_state_probe.sh

echo "==================== diagnostic smoke complete ======================="
