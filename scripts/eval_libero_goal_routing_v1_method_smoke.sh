#!/usr/bin/env bash
set -euo pipefail

# Recommended method smoke before 50-trial runs:
# B1 + B2, CL-only only, all three score definitions, 1 trial/task by default.
# This tests actual multi-expert discrimination without paying for Base-inclusive twice yet.
MODES="b1 b2" \
PROTOCOLS="cl_only" \
SCORES="world latent combined" \
NUM_TRIALS="${NUM_TRIALS:-1}" \
EVAL_WORKERS="${EVAL_WORKERS:-4}" \
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}" \
GROUP_STAMP="${GROUP_STAMP:-$(date +"%Y%m%d_%H%M%S")_method_smoke}" \
bash scripts/eval_libero_goal_routing_v1_grid.sh
