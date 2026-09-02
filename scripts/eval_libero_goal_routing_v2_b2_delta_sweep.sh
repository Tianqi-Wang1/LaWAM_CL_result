#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
GROUP_STAMP="${GROUP_STAMP:-$(date +"%Y%m%d_%H%M%S")_b2_delta_sweep}"
DELTAS="${DELTAS:-0.20 0.10}"

roots=()
for delta in ${DELTAS}; do
  tag="${delta/./p}"
  echo "######################################################################"
  echo " B2 closed-loop delta=${delta}"
  echo "######################################################################"
  ROOT="${ROOT}" V2_ROOT="${V2_ROOT}" \
  GATE_THRESHOLD="${delta}" EVAL_STAMP="${GROUP_STAMP}_${tag}" \
  bash scripts/eval_libero_goal_routing_v2_b2_closed_loop_cl.sh
  pointer="${V2_ROOT}/latest_b2_closed_loop_delta_${tag}.txt"
  [ -f "${pointer}" ] || { echo "[ERROR] missing pointer ${pointer}"; exit 1; }
  roots+=("${delta}=$(cat "${pointer}")")
done

COMPARE_DIR="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_b2_closed_loop_cl/delta_comparisons/${GROUP_STAMP}"
args=()
for item in "${roots[@]}"; do args+=(--run "${item}"); done
python scripts/compare_routing_v2_b2_delta_runs.py "${args[@]}" --output-dir "${COMPARE_DIR}"
echo "${COMPARE_DIR}" > "${V2_ROOT}/latest_b2_delta_comparison.txt"
echo "[OK] B2 delta sweep complete -> ${COMPARE_DIR}"
