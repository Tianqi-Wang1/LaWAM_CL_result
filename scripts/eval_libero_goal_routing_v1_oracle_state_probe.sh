#!/usr/bin/env bash
set -euo pipefail

# Oracle-state diagnostic:
#   1) all T6-T9 candidates are scored WITHOUT task ID entering model features/scores;
#   2) after scores are computed, GT task metadata is used ONLY to execute the correct expert;
#   3) this prevents early routing mistakes from pushing the robot off-distribution;
#   4) raw dz/dh logs are re-scored offline as latent/world/raw-combined/normalized-combined.
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

NUM_TRIALS="${NUM_TRIALS:-3}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
MODES="${MODES:-b1 b2}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")_oracle_state_probe}"

for mode in ${MODES}; do
  echo "######################################################################"
  echo " Oracle-state routing-score probe | mode=${mode} | final bank T6-T9 | trials=${NUM_TRIALS}/task"
  echo " Scores are task-agnostic; GT is used only after scoring to choose executed action."
  echo "######################################################################"
  MODE="${mode}" \
  PROTOCOL=cl_only \
  SCORE_MODE=latent \
  NUM_TRIALS="${NUM_TRIALS}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  STAGES=CL4 \
  PROTOCOL_POSTPROCESS=false \
  ROUTING_EXECUTION_MODE=oracle_execute \
  ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-8}" \
  EVAL_STAMP="${STAMP}" \
  bash scripts/eval_libero_goal_routing_v1_bank.sh

  OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_no_taskid/${mode}/cl_only/latent_a1.0/${STAMP}"
  LOG="${OUT_ROOT}/routing_chunks_all_stages.jsonl"
  [ -s "${LOG}" ] || { echo "[ERROR] oracle probe log missing: ${LOG}" >&2; exit 1; }
  /home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python scripts/analyze_routing_v1_oracle_probe.py \
    --log "${LOG}" \
    --output-dir "${OUT_ROOT}/oracle_probe_analysis" \
    --combined-alpha 0.5

done
