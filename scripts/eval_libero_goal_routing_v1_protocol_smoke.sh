#!/usr/bin/env bash
set -euo pipefail

# Wiring smoke across B1/B2 and CL-only/Base-inclusive for ONE score mode.
# Example:
#   SCORE_MODE=world NUM_TRIALS=1 bash scripts/eval_libero_goal_routing_v1_protocol_smoke.sh
#   SCORE_MODE=combined ALPHA=0.5 NUM_TRIALS=1 bash scripts/eval_libero_goal_routing_v1_protocol_smoke.sh
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

NUM_TRIALS="${NUM_TRIALS:-1}"
SCORE_MODE="${SCORE_MODE:-world}"
ALPHA="${ALPHA:-}"
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}"
GROUP_STAMP="${GROUP_STAMP:-$(date +"%Y%m%d_%H%M%S")_protocol_smoke}"

for mode in b1 b2; do
  for protocol in cl_only base_inclusive; do
    echo "======================================================================"
    echo " Routing-V1 protocol smoke: mode=${mode} protocol=${protocol} score=${SCORE_MODE} trials=${NUM_TRIALS}"
    echo "======================================================================"
    MODE="${mode}" \
    PROTOCOL="${protocol}" \
    SCORE_MODE="${SCORE_MODE}" \
    ALPHA="${ALPHA}" \
    NUM_TRIALS="${NUM_TRIALS}" \
    ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS}" \
    EVAL_STAMP="${GROUP_STAMP}" \
    bash scripts/eval_libero_goal_routing_v1_bank.sh
  done
done
