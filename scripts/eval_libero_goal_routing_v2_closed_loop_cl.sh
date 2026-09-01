#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DEBUG_DECISIONS="${DEBUG_DECISIONS:-8}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.20}"
LAMBDA_MAX="${LAMBDA_MAX:-0.50}"
FUSION_GAMMA="${FUSION_GAMMA:-2.0}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"
OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_closed_loop_cl/${STAMP}"
mkdir -p "${OUT_ROOT}"

# Fail early if any formal Skill/Memory pointer is missing.
for t in 6 7 8 9; do
  [ -f "${V2_ROOT}/task${t}/latest_skill_run.txt" ] || { echo "[ERROR] missing latest_skill_run for T${t}"; exit 1; }
  run=$(cat "${V2_ROOT}/task${t}/latest_skill_run.txt")
  [ -f "${run}/final_model/pytorch_model.pt" ] || { echo "[ERROR] missing skill checkpoint T${t}"; exit 1; }
  [ -f "${V2_ROOT}/task${t}/routing_memory/routing_memory.pt" ] || { echo "[ERROR] missing routing memory T${t}"; exit 1; }
done

cat > "${OUT_ROOT}/PROTOCOL.txt" <<EOF
Routing-V2 formal CLOSED-LOOP CL-only evaluation.
No task ID is used by selection. Each task is evaluated in a separate process only so GT can be logged diagnostically.
CL1 bank: T6; eval T6.
CL2 bank: T6,T7; eval T6,T7.
CL3 bank: T6,T7,T8; eval T6,T7,T8.
CL4 bank: T6,T7,T8,T9; eval T6,T7,T8,T9.
Per action chunk:
  Base Semantic AE retrieval -> C_sem=(e2-e1)/(e2+eps).
  Gate if C_sem < ${GATE_THRESHOLD}.
  If gated, Top-2 candidate WAM imagination + Dynamics AE and adaptive pair-normalized fusion.
  lambda_max=${LAMBDA_MAX}, gamma=${FUSION_GAMMA}.
The selected Skill Path generates the real action chunk.
NUM_TRIALS=${NUM_TRIALS}
EOF

echo "======================================================================"
echo " Routing-V2 FORMAL CLOSED-LOOP CL evaluation"
echo " Gate/fusion : delta=${GATE_THRESHOLD}, lambda_max=${LAMBDA_MAX}, gamma=${FUSION_GAMMA}"
echo " Trials      : ${NUM_TRIALS} / task / stage"
echo " Output      : ${OUT_ROOT}"
echo "======================================================================"

run_stage() {
  local stage="$1" candidates="$2" tasks="$3" port_seed="$4"
  local stage_dir="${OUT_ROOT}/${stage}"
  mkdir -p "${stage_dir}"
  echo "######################################################################"
  echo " ${stage}: bank={${candidates}} eval={${tasks}}"
  echo "######################################################################"
  local offset=0
  for task in ${tasks}; do
    STAGE="${stage}" TASK_ID="${task}" CANDIDATE_TASKS="${candidates}" \
    OUTPUT_ROOT="${stage_dir}" V2_ROOT="${V2_ROOT}" NUM_TRIALS="${NUM_TRIALS}" \
    EVAL_WORKERS="${EVAL_WORKERS}" POLICY_GPU="${POLICY_GPU}" EVAL_GPU="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" DEBUG_DECISIONS="${DEBUG_DECISIONS}" \
    GATE_THRESHOLD="${GATE_THRESHOLD}" LAMBDA_MAX="${LAMBDA_MAX}" FUSION_GAMMA="${FUSION_GAMMA}" \
    PORT_BASE="$((port_seed + offset*20))" \
    bash scripts/run_libero_goal_routing_v2_closed_loop_task.sh
    offset=$((offset+1))
  done
  python scripts/combine_routing_v2_stage_summaries.py \
    --stage-dir "${stage_dir}" --tasks ${tasks} --output "${stage_dir}/per_task_summary.csv"
  python scripts/summarize_routing_v2_closed_loop.py \
    --stage-dir "${stage_dir}" --tasks ${tasks} --output-dir "${stage_dir}/routing_summary"
}

run_stage CL1 "6"       "6"       6200
run_stage CL2 "6 7"     "6 7"     6300
run_stage CL3 "6 7 8"   "6 7 8"   6400
run_stage CL4 "6 7 8 9" "6 7 8 9" 6500

MATRIX="${OUT_ROOT}/sr_matrix_cl_only.csv"
python scripts/build_routing_v2_sr_matrix.py --root "${OUT_ROOT}" --output "${MATRIX}"
python scripts/compute_routing_v2_cl_metrics.py --matrix "${MATRIX}" --output-dir "${OUT_ROOT}/metrics"

# Compact cross-stage routing table.
python - "${OUT_ROOT}" <<'PY'
import csv,sys
from pathlib import Path
root=Path(sys.argv[1]); out=root/'routing_summary_all_stages.csv'
rows=[]
for stage in ['CL1','CL2','CL3','CL4']:
    p=root/stage/'routing_summary'/'routing_summary.csv'
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        allrow=next(r for r in csv.DictReader(f) if r['scope']=='ALL')
    rows.append({'stage':stage,**allrow})
with out.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['stage','scope','decisions','semantic_top1_accuracy','routing_accuracy','gate_activation_rate','mean_lambda_dyn']); w.writeheader(); w.writerows(rows)
print('[RoutingV2] all-stage routing summary')
print(out.read_text(encoding='utf-8').rstrip())
PY

echo "${OUT_ROOT}" > "${V2_ROOT}/latest_closed_loop_cl_run.txt"
echo "======================================================================"
echo " Routing-V2 CLOSED-LOOP CL evaluation COMPLETE"
echo " Output  : ${OUT_ROOT}"
echo " Matrix  : ${MATRIX}"
echo " Metrics : ${OUT_ROOT}/metrics/metrics.json"
echo " Routing : ${OUT_ROOT}/routing_summary_all_stages.csv"
echo "======================================================================"
