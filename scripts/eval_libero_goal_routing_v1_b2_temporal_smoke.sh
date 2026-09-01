#!/usr/bin/env bash
set -euo pipefail

# B2-only closed-loop temporal routing smoke.
# Keeps one routing decision per action chunk; temporal stabilization only smooths
# branch-wise normalized self-consistency scores over chunks of the SAME episode.
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

NUM_TRIALS="${NUM_TRIALS:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
ALPHA="${ALPHA:-0.8}"
GROUP_STAMP="${GROUP_STAMP:-$(date +"%Y%m%d_%H%M%S")_b2_temporal_smoke}"
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}"

run_one() {
  local temporal_mode="$1" beta="$2" margin="$3"
  echo "######################################################################"
  echo " B2 temporal smoke: mode=${temporal_mode} beta=${beta} margin=${margin}"
  echo " score=combined alpha=${ALPHA} norm=candidate_mean | stage=CL4"
  echo " one routing decision is STILL made per action chunk"
  echo "######################################################################"
  MODE=b2 \
  PROTOCOL=cl_only \
  SCORE_MODE=combined \
  ALPHA="${ALPHA}" \
  SCORE_NORMALIZATION=candidate_mean \
  TEMPORAL_MODE="${temporal_mode}" \
  TEMPORAL_BETA="${beta}" \
  TEMPORAL_MARGIN="${margin}" \
  STAGES=CL4 \
  PROTOCOL_POSTPROCESS=false \
  NUM_TRIALS="${NUM_TRIALS}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS}" \
  EVAL_STAMP="${GROUP_STAMP}" \
  bash scripts/eval_libero_goal_routing_v1_bank.sh
}

# Baseline: normalized 0.8/0.2 fusion but no temporal memory.
run_one none 0.0 0.0
# Light EMA selected from the clean-state probe as the first candidate.
run_one ema 0.3 0.0
# Same EMA plus a very small switch threshold.
run_one ema_hysteresis 0.3 0.08

echo "======================================================================"
echo "B2 temporal smoke finished. Compare CL4 routing_per_context.csv and per_task_summary.csv."
echo "Look for: final routing accuracy ↑, switch rate ↓, and SR not decreasing."
echo "======================================================================"
