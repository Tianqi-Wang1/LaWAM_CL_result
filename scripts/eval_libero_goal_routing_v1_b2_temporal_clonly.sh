#!/usr/bin/env bash
set -euo pipefail

# B2-only CL-only Routing-V1 with normalized semantic-motor/world self-consistency
# and lightweight per-episode EMA stabilization. Routing is still recomputed once
# per action chunk. Defaults are diagnostic; override NUM_TRIALS=50 for formal eval.
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

MODE=b2 \
PROTOCOL=cl_only \
SCORE_MODE=combined \
ALPHA="${ALPHA:-0.8}" \
SCORE_NORMALIZATION="${SCORE_NORMALIZATION:-candidate_mean}" \
TEMPORAL_MODE="${TEMPORAL_MODE:-ema}" \
TEMPORAL_BETA="${TEMPORAL_BETA:-0.3}" \
TEMPORAL_MARGIN="${TEMPORAL_MARGIN:-0.0}" \
STAGES="${STAGES:-all}" \
PROTOCOL_POSTPROCESS="${PROTOCOL_POSTPROCESS:-auto}" \
NUM_TRIALS="${NUM_TRIALS:-10}" \
EVAL_WORKERS="${EVAL_WORKERS:-8}" \
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}" \
bash scripts/eval_libero_goal_routing_v1_bank.sh
