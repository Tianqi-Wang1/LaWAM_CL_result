#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam
ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
NUM_TRIALS="${NUM_TRIALS:-10}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DEBUG_DECISIONS="${DEBUG_DECISIONS:-4}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.20}"
LAMBDA_MAX="${LAMBDA_MAX:-0.50}"
FUSION_GAMMA="${FUSION_GAMMA:-2.0}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_dynamics_2x3_probe/${STAMP}"
mkdir -p "${OUT_ROOT}"

VARIANTS=(taskwm_hdhz taskwm_hdh taskwm_dh basewm_hdhz basewm_hdh basewm_dh)

# Preflight: all 24 Dynamics memories + 4 Semantic memories + 4 upstream deltas.
for t in 6 7 8 9; do
  [ -f "${V2_ROOT}/task${t}/routing_memory/routing_memory.pt" ] || { echo "[ERROR] missing original memory T${t}"; exit 1; }
  [ -f "${V2_ROOT}/task${t}/routing_upstream_delta/routing_upstream_delta.pt" ] || { echo "[ERROR] missing upstream delta T${t}"; exit 1; }
  for v in "${VARIANTS[@]}"; do
    p="${V2_ROOT}/task${t}/routing_memory_variants/${v}/dynamics_ae.pt"
    [ -f "${p}" ] || { echo "[ERROR] missing ${v}/T${t}: ${p}"; exit 1; }
  done
done

cat > "${OUT_ROOT}/PROTOCOL.txt" <<EOF
Routing-V2 paired 2x3 passive future-dynamics ablation.
CL-only bank: T6,T7,T8,T9.
The six variants are evaluated on EXACTLY the same rollout chunks in a single pass.
Robot action always uses the provided oracle/task-ID Skill; routing never controls action.
Semantic Top-2 is shared across all variants.
For each Top-2 candidate, task-WM and shared-Base-WM futures are each computed once.
Dynamics inputs:
  taskwm_hdhz = Task-WM + [h,Delta h,z]
  taskwm_hdh  = Task-WM + [h,Delta h]
  taskwm_dh   = Task-WM + [Delta h]
  basewm_hdhz = Base-WM + [h,Delta h,z]
  basewm_hdh  = Base-WM + [h,Delta h]
  basewm_dh   = Base-WM + [Delta h]
Fixed comparison fusion hyperparameters (NO per-variant tuning):
  delta=${GATE_THRESHOLD}, lambda_max=${LAMBDA_MAX}, gamma=${FUSION_GAMMA}
EOF

echo "======================================================================"
echo " Routing-V2 PAIRED 2x3 passive probe"
echo " Bank        : T6 T7 T8 T9"
echo " Variants    : ${VARIANTS[*]}"
echo " Trials      : ${NUM_TRIALS}/task"
echo " Workers     : ${EVAL_WORKERS}"
echo " Oracle action: YES (probe only)"
echo " Fusion eval : delta=${GATE_THRESHOLD} lambda_max=${LAMBDA_MAX} gamma=${FUSION_GAMMA}"
echo " Output      : ${OUT_ROOT}"
echo "======================================================================"

for t in 6 7 8 9; do
  TASK_ID="${t}" \
  V2_ROOT="${V2_ROOT}" \
  NUM_TRIALS="${NUM_TRIALS}" \
  EVAL_WORKERS="${EVAL_WORKERS}" \
  POLICY_GPU="${POLICY_GPU}" \
  EVAL_GPU="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" \
  DEBUG_DECISIONS="${DEBUG_DECISIONS}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  PORT_BASE="$((6894+t*10))" \
  bash scripts/run_libero_goal_routing_v2_2x3_probe_task.sh
done

python scripts/summarize_routing_v2_2x3_probe.py \
  --root "${OUT_ROOT}" \
  --gate-threshold "${GATE_THRESHOLD}" \
  --lambda-max "${LAMBDA_MAX}" \
  --gamma "${FUSION_GAMMA}"

echo "${OUT_ROOT}" > "${V2_ROOT}/latest_dynamics_2x3_probe_run.txt"
echo "[OK] Routing-V2 paired 2x3 passive probe complete: ${OUT_ROOT}"
