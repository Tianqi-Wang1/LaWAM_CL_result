#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam
ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"; cd "${ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
NUM_TRIALS="${NUM_TRIALS:-5}"; EVAL_WORKERS="${EVAL_WORKERS:-4}"; POLICY_GPU="${POLICY_GPU:-4}"; EVAL_GPU="${EVAL_GPU:-5}"; SAVE_VIDEOS="${SAVE_VIDEOS:-False}"; DEBUG_DECISIONS="${DEBUG_DECISIONS:-8}"
STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"; OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_dynamics_probe/${STAMP}"; mkdir -p "${OUT_ROOT}"

# Extract and cache the only candidate-specific policy tensors needed by Stage 2.
for t in 6 7 8 9; do
  run=$(cat "${V2_ROOT}/task${t}/latest_skill_run.txt"); ckpt="${run}/final_model/pytorch_model.pt"; [ -f "${ckpt}" ] || { echo "[ERROR] missing T${t} skill ${ckpt}"; exit 1; }
  mem="${V2_ROOT}/task${t}/routing_memory/routing_memory.pt"; [ -f "${mem}" ] || { echo "[ERROR] missing T${t} memory"; exit 1; }
  delta_dir="${V2_ROOT}/task${t}/routing_upstream_delta"; delta="${delta_dir}/routing_upstream_delta.pt"
  if [ ! -f "${delta}" ] || [ "${FORCE_EXTRACT_DELTA:-False}" = "True" ]; then
    mkdir -p "${delta_dir}"; python scripts/extract_routing_v2_upstream_delta.py --checkpoint "${ckpt}" --output "${delta}"
  else
    echo "[RoutingV2] reuse T${t} upstream delta: ${delta}"
  fi
done

cat >"${OUT_ROOT}/PROTOCOL.txt" <<EOF
Routing-V2 passive Semantic->Dynamics verification, CL-only bank T6-T9.
Robot execution always uses provided GT/task-ID Skill Path.
At every action chunk:
1) Base VLM + Base queries -> common H_act anchor.
2) All Semantic AEs score anchor, produce Top-2.
3) Only Top-2 task upstream deltas (VLM Text-LoRA + query residual + QFormer-LoRA + LaWM-LoRA) are activated sequentially.
4) Each candidate predicts z_k and future world feature h_k; its own Dynamics AE reconstructs that predicted transition.
5) Dynamics winner is logged but NEVER controls action.
6) GT upstream delta is restored before oracle action generation.
Outputs support Recovery/Damage and confidence-gated hybrid analysis.
EOF

echo "======================================================================"
echo " Routing-V2 passive Dynamics bank probe"
echo " Bank      : T6 T7 T8 T9"
echo " Trials    : ${NUM_TRIALS}/task"
echo " Action    : oracle task-ID only"
echo " Output    : ${OUT_ROOT}"
echo "======================================================================"
for t in 6 7 8 9; do
  TASK_ID="${t}" V2_ROOT="${V2_ROOT}" NUM_TRIALS="${NUM_TRIALS}" EVAL_WORKERS="${EVAL_WORKERS}" POLICY_GPU="${POLICY_GPU}" EVAL_GPU="${EVAL_GPU}" SAVE_VIDEOS="${SAVE_VIDEOS}" DEBUG_DECISIONS="${DEBUG_DECISIONS}" OUTPUT_ROOT="${OUT_ROOT}" PORT_BASE="$((5894+t*10))" bash scripts/run_libero_goal_routing_v2_dynamics_probe_task.sh
done
python scripts/summarize_routing_v2_dynamics_probe.py --root "${OUT_ROOT}"
echo "${OUT_ROOT}" >"${V2_ROOT}/latest_dynamics_probe_run.txt"
echo "[OK] Routing-V2 Dynamics probe complete: ${OUT_ROOT}"
