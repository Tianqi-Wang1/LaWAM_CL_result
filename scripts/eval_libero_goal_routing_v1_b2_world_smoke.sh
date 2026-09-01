#!/usr/bin/env bash
set -euo pipefail
# B2 / CL-only / world-only smoke. Each candidate activates its own Text-LoRA,
# obtains z_k/h_k, and compares against the same branch's z*_k/h*_k.
MODE=b2 \
PROTOCOL="${PROTOCOL:-cl_only}" \
SCORE_MODE=world \
NUM_TRIALS="${NUM_TRIALS:-2}" \
EVAL_WORKERS="${EVAL_WORKERS:-4}" \
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}" \
bash scripts/eval_libero_goal_routing_v1_bank.sh
