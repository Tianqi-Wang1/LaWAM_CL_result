#!/usr/bin/env bash
set -euo pipefail
# B1 / CL-only / world-only smoke. Base expert is absent from competition.
MODE=b1 \
PROTOCOL="${PROTOCOL:-cl_only}" \
SCORE_MODE=world \
NUM_TRIALS="${NUM_TRIALS:-2}" \
EVAL_WORKERS="${EVAL_WORKERS:-4}" \
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}" \
bash scripts/eval_libero_goal_routing_v1_bank.sh
