#!/usr/bin/env bash
set -euo pipefail

# Serial Routing-V1 evaluation grid.
# Defaults = full experiment group:
#   B1/B2 x CL-only/Base-inclusive x world/latent/combined = 12 runs.
# The Base acquisition row is evaluated once and automatically reused across
# subsequent Base-inclusive runs, because it has no routing competition.
# Override examples:
#   NUM_TRIALS=1 PROTOCOLS="cl_only" bash scripts/eval_libero_goal_routing_v1_grid.sh
#   MODES="b1" PROTOCOLS="cl_only" SCORES="world latent combined" NUM_TRIALS=50 bash ...

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

MODES="${MODES:-b1 b2}"
PROTOCOLS="${PROTOCOLS:-cl_only base_inclusive}"
SCORES="${SCORES:-world latent combined}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-5}"
COMBINED_ALPHA="${COMBINED_ALPHA:-0.5}"
GROUP_STAMP="${GROUP_STAMP:-$(date +"%Y%m%d_%H%M%S")_grid}"
SHARED_BASE_SUMMARY="${SHARED_BASE_SUMMARY:-}"

for mode in ${MODES}; do
  for protocol in ${PROTOCOLS}; do
    for score in ${SCORES}; do
      echo "######################################################################"
      echo " Routing grid: mode=${mode} protocol=${protocol} score=${score} trials=${NUM_TRIALS}"
      echo " group stamp : ${GROUP_STAMP}"
      if [ "${protocol}" = "base_inclusive" ] && [ -n "${SHARED_BASE_SUMMARY}" ]; then
        echo " Base row    : REUSE ${SHARED_BASE_SUMMARY}"
      fi
      echo "######################################################################"
      if [ "${score}" = "combined" ]; then
        alpha="${COMBINED_ALPHA}"
        canonical_alpha="${COMBINED_ALPHA}"
      elif [ "${score}" = "world" ]; then
        alpha=""
        canonical_alpha="0.0"
      else
        alpha=""
        canonical_alpha="1.0"
      fi

      MODE="${mode}" \
      PROTOCOL="${protocol}" \
      SCORE_MODE="${score}" \
      ALPHA="${alpha}" \
      NUM_TRIALS="${NUM_TRIALS}" \
      EVAL_WORKERS="${EVAL_WORKERS}" \
      POLICY_GPU="${POLICY_GPU}" \
      EVAL_GPU="${EVAL_GPU}" \
      ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS}" \
      BASE_SUMMARY_CSV_OVERRIDE="${SHARED_BASE_SUMMARY}" \
      EVAL_STAMP="${GROUP_STAMP}" \
      bash scripts/eval_libero_goal_routing_v1_bank.sh

      # Cache the first Base acquisition summary and reuse it for every later
      # Base-inclusive score/mode. This avoids repeated identical Base rollouts.
      if [ "${protocol}" = "base_inclusive" ] && [ -z "${SHARED_BASE_SUMMARY}" ]; then
        run_root="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_no_taskid/${mode}/${protocol}/${score}_a${canonical_alpha}/${GROUP_STAMP}"
        SHARED_BASE_SUMMARY=$(find "${run_root}/Base" -type f -name per_task_summary.csv 2>/dev/null | sort | tail -n 1)
        [ -n "${SHARED_BASE_SUMMARY}" ] || { echo "[ERROR] Failed to locate reusable Base summary under ${run_root}/Base"; exit 1; }
        echo "[RoutingV1] Cached shared Base summary: ${SHARED_BASE_SUMMARY}"
      fi
    done
  done
done

echo "[RoutingV1] Grid complete: ${GROUP_STAMP}"
