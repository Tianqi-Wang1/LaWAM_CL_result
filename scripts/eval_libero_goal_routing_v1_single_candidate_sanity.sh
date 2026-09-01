#!/usr/bin/env bash
set -euo pipefail

# Sanity check: with a single candidate (T6), Routing-V1 must reduce to the
# ordinary oracle T6 expert. This checks functional parameter swapping, common
# noise, action serialization, and the closed-loop evaluator without routing ambiguity.
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

NUM_TRIALS="${NUM_TRIALS:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
MODES="${MODES:-b1 b2}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")_single_candidate_sanity}"

for mode in ${MODES}; do
  echo "######################################################################"
  echo " Single-candidate sanity | mode=${mode} | T6 only | trials=${NUM_TRIALS}"
  echo " Expected: SR should approach the corresponding oracle T6 SR; routing acc=1.0."
  echo "######################################################################"
  MODE="${mode}" \
  PROTOCOL=cl_only \
  SCORE_MODE=latent \
  NUM_TRIALS="${NUM_TRIALS}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  STAGES=CL1 \
  PROTOCOL_POSTPROCESS=false \
  ROUTING_EXECUTION_MODE=route \
  EVAL_STAMP="${STAMP}" \
  bash scripts/eval_libero_goal_routing_v1_bank.sh
done
