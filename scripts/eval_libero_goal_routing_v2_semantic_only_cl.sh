#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DEBUG_DECISIONS="${DEBUG_DECISIONS:-0}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}" 
OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_semantic_only_closed_loop_cl/${STAMP}"
mkdir -p "${OUT_ROOT}"

# Semantic-only is implemented with the already validated B2 closed-loop server,
# but the confidence gate is hard-disabled.  Because semantic errors are sorted
# before confidence is computed, C_sem >= 0 by construction; with delta=0 no
# sample can enter Stage-2.  Hence no Base-WM future or Dynamics AE forward is
# executed at inference time.  The Dynamics checkpoints are merely loaded by the
# existing server startup path and never used for selection.
readonly GATE_THRESHOLD="0.0"
readonly LAMBDA_MAX="0.0"
readonly FUSION_GAMMA="2.0"
export GATE_THRESHOLD LAMBDA_MAX FUSION_GAMMA

required=(
  scripts/run_libero_goal_routing_v2_b2_closed_loop_task.sh
  scripts/combine_routing_v2_stage_summaries.py
  scripts/summarize_routing_v2_b2_closed_loop.py
  scripts/build_routing_v2_sr_matrix.py
  scripts/compute_routing_v2_cl_metrics.py
)
for p in "${required[@]}"; do
  [ -f "${p}" ] || { echo "[ERROR] missing required B2 closed-loop component: ${p}"; exit 1; }
done

for t in 6 7 8 9; do
  [ -f "${V2_ROOT}/task${t}/latest_skill_run.txt" ] || { echo "[ERROR] missing latest_skill_run for T${t}"; exit 1; }
  run=$(cat "${V2_ROOT}/task${t}/latest_skill_run.txt")
  [ -f "${run}/final_model/pytorch_model.pt" ] || { echo "[ERROR] missing skill checkpoint T${t}"; exit 1; }
  [ -f "${V2_ROOT}/task${t}/routing_memory/routing_memory.pt" ] || { echo "[ERROR] missing Semantic memory T${t}"; exit 1; }
  # The current B2 server loader still expects these files at startup even
  # though semantic-only routing never performs a Dynamics forward.
  [ -f "${V2_ROOT}/task${t}/routing_memory_variants/basewm_hdh/dynamics_ae.pt" ] || { echo "[ERROR] missing B2 Dynamics memory T${t} required by current server loader"; exit 1; }
done

cat > "${OUT_ROOT}/PROTOCOL.txt" <<EOF2
Routing-V2 SEMANTIC-ONLY formal CLOSED-LOOP CL-only evaluation.
Selection = shared Base-VLM Semantic AE Top-1 ONLY.
No task ID is used by routing selection. GT task is available only to the environment and diagnostics.
CL1 bank: T6; CL2: T6,T7; CL3: T6,T7,T8; CL4: T6,T7,T8,T9.
Dynamics / Base-WM verification is disabled by construction:
  gate threshold delta = 0.0
  lambda_max = 0.0
Since Semantic errors are sorted (e1 <= e2), C_sem=(e2-e1)/(e2+eps) >= 0, so gate_active is always False.
Selected Semantic Top-1 Skill executes with its COMPLETE task-specific path, including its task-specific LaWM-LoRA and Action Expert.
NUM_TRIALS=${NUM_TRIALS}
EOF2

echo "======================================================================"
echo " Routing-V2 SEMANTIC-ONLY FORMAL CLOSED-LOOP CL evaluation"
echo " Router        : Base Semantic AE Top-1 only"
echo " Dynamics      : DISABLED (delta=0, lambda_max=0)"
echo " GT task ID    : DIAGNOSTICS ONLY"
echo " Trials        : ${NUM_TRIALS} / task / stage"
echo " Workers       : ${EVAL_WORKERS}"
echo " Output        : ${OUT_ROOT}"
echo "======================================================================"

run_stage() {
  local stage="$1" candidates="$2" tasks="$3" port_seed="$4"
  local stage_dir="${OUT_ROOT}/${stage}"
  mkdir -p "${stage_dir}"
  echo "######################################################################"
  echo " ${stage}: semantic bank={${candidates}} eval={${tasks}}"
  echo "######################################################################"
  local offset=0
  for task in ${tasks}; do
    STAGE="${stage}" TASK_ID="${task}" CANDIDATE_TASKS="${candidates}" \
    OUTPUT_ROOT="${stage_dir}" V2_ROOT="${V2_ROOT}" NUM_TRIALS="${NUM_TRIALS}" \
    EVAL_WORKERS="${EVAL_WORKERS}" POLICY_GPU="${POLICY_GPU}" EVAL_GPU="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" DEBUG_DECISIONS="${DEBUG_DECISIONS}" \
    WORKER_RESULT_TIMEOUT_SEC="${WORKER_RESULT_TIMEOUT_SEC:-1800}" \
    SERVER_STARTUP_TIMEOUT_SEC="${SERVER_STARTUP_TIMEOUT_SEC:-1200}" \
    PORT_BASE="$((port_seed + offset*20))" \
    bash scripts/run_libero_goal_routing_v2_b2_closed_loop_task.sh
    offset=$((offset+1))
  done

  python scripts/combine_routing_v2_stage_summaries.py \
    --stage-dir "${stage_dir}" --tasks ${tasks} --output "${stage_dir}/per_task_summary.csv"
  python scripts/summarize_routing_v2_b2_closed_loop.py \
    --stage-dir "${stage_dir}" --tasks ${tasks} --output-dir "${stage_dir}/routing_summary"
}

run_stage CL1 "6"       "6"       7600
run_stage CL2 "6 7"     "6 7"     7700
run_stage CL3 "6 7 8"   "6 7 8"   7800
run_stage CL4 "6 7 8 9" "6 7 8 9" 7900

MATRIX="${OUT_ROOT}/sr_matrix_cl_only.csv"
python scripts/build_routing_v2_sr_matrix.py --root "${OUT_ROOT}" --output "${MATRIX}"
python scripts/compute_routing_v2_cl_metrics.py --matrix "${MATRIX}" --output-dir "${OUT_ROOT}/metrics"

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
fields=['stage']+[k for k in rows[0] if k!='stage']
with out.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print('[RoutingV2][SemanticOnly] all-stage routing summary')
print(out.read_text(encoding='utf-8').rstrip())
PY

python scripts/validate_routing_v2_semantic_only.py --root "${OUT_ROOT}"

echo "${OUT_ROOT}" > "${V2_ROOT}/latest_semantic_only_closed_loop_cl.txt"

echo "======================================================================"
echo " Routing-V2 SEMANTIC-ONLY CLOSED-LOOP CL evaluation COMPLETE"
echo " Output     : ${OUT_ROOT}"
echo " Matrix     : ${MATRIX}"
echo " Metrics    : ${OUT_ROOT}/metrics/metrics.json"
echo " Routing    : ${OUT_ROOT}/routing_summary_all_stages.csv"
echo " Validation : ${OUT_ROOT}/semantic_only_validation.json"
echo " Pointer    : ${V2_ROOT}/latest_semantic_only_closed_loop_cl.txt"
echo "======================================================================"
