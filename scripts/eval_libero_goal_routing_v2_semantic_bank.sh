#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam
ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"; cd "${ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
NUM_TRIALS="${NUM_TRIALS:-5}"; EVAL_WORKERS="${EVAL_WORKERS:-4}"; POLICY_GPU="${POLICY_GPU:-4}"; EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"; DEBUG_DECISIONS="${DEBUG_DECISIONS:-8}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"; OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_semantic_probe/${STAMP}"
mkdir -p "${OUT_ROOT}"

for t in 6 7 8 9; do
  [ -f "${V2_ROOT}/task${t}/latest_skill_run.txt" ] || { echo "[ERROR] missing T${t} latest_skill_run.txt"; exit 1; }
  [ -f "${V2_ROOT}/task${t}/routing_memory/routing_memory.pt" ] || { echo "[ERROR] missing T${t} routing_memory.pt"; exit 1; }
done

cat > "${OUT_ROOT}/PROTOCOL.txt" <<EOF
Routing-V2 Semantic AE Bank passive validation.
Candidate bank: T6,T7,T8,T9 only (CL-only).
At every action-chunk query:
  1) compute Base-VLM + Base-query H_act (task VLM-LoRA disabled; task query delta unused),
  2) score the SAME H_act with all four task Semantic AEs,
  3) record raw reconstruction-error ranking and GT rank.
The semantic ranking NEVER controls the robot.
Each task rollout executes its provided task-ID V2 skill checkpoint.
Primary metrics: chunk-level Top-1 accuracy and Top-2 recall.
EOF

echo "======================================================================"
echo " Routing-V2 Semantic AE Bank passive validation"
echo " Bank      : T6 T7 T8 T9"
echo " Trials    : ${NUM_TRIALS}/task"
echo " Execution : provided task ID; semantic result cannot control robot"
echo " Output    : ${OUT_ROOT}"
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
  PORT_BASE="$((5794 + t * 10))" \
  bash scripts/run_libero_goal_routing_v2_semantic_probe_task.sh
done

python scripts/summarize_routing_v2_semantic_probe.py --root "${OUT_ROOT}"

echo "${OUT_ROOT}" > "${V2_ROOT}/latest_semantic_probe_run.txt"
echo "[OK] Routing-V2 Semantic AE Bank probe complete: ${OUT_ROOT}"
