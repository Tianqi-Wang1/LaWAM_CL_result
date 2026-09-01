#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

ANALYSIS_SCRIPT="${ROOT}/scripts/analyze_lawam_full_flow_update_rank.py"

BASE_CKPT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft/20260807_124027+base_t0_5_10k_4gpu_bs32_ga2/final_model/pytorch_model.pt"

T6_CKPT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_head_only_oracle/heads/20260819_102751+cl1_t6_2k_4gpu_bs32_ga2_flow_head_only_from_base/final_model/pytorch_model.pt"
T7_CKPT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_head_only_oracle/heads/20260819_113303+cl2_t7_2k_4gpu_bs32_ga2_flow_head_only_from_base/final_model/pytorch_model.pt"
T8_CKPT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_head_only_oracle/heads/20260819_124003+cl3_t8_2k_4gpu_bs32_ga2_flow_head_only_from_base/final_model/pytorch_model.pt"
T9_CKPT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_head_only_oracle/heads/20260819_134644+cl4_t9_2k_4gpu_bs32_ga2_flow_head_only_from_base/final_model/pytorch_model.pt"

for required in \
    "${ANALYSIS_SCRIPT}" \
    "${BASE_CKPT}" \
    "${T6_CKPT}" \
    "${T7_CKPT}" \
    "${T8_CKPT}" \
    "${T9_CKPT}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Missing required file:"
        echo "        ${required}"
        exit 1
    fi
done

# One GPU is enough; SVD is performed sequentially, one Delta-W matrix at a time.
# ANALYSIS_GPU is a PHYSICAL GPU id. After CUDA_VISIBLE_DEVICES remapping, Python uses cuda:0.
ANALYSIS_GPU="${ANALYSIS_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_update_rank/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/analysis.log"

cat > "${OUTPUT_DIR}/lora_sr_reference.csv" <<'CSV'
task,task_id,flow_lora_r8_sr
T6,6,0.70
T7,7,0.98
T8,8,0.86
T9,9,0.34
CSV

cat > "${OUTPUT_DIR}/PROTOCOL.txt" <<EOF
Goal Full-Flow update localization and intrinsic-rank analysis.

Formal Base:
  ${BASE_CKPT}

Independent Full-Flow task checkpoints:
  T6: ${T6_CKPT}
  T7: ${T7_CKPT}
  T8: ${T8_CKPT}
  T9: ${T9_CKPT}

Current Flow-LoRA v1 targets:
  16 Transformer blocks x 6 Linear weights = 96 target matrices
  Q/K/V/O + FFN-in + FFN-out

Analysis:
  1. Compute Delta theta = theta_task - theta_Base for every canonical
     policy_backend.flow.* tensor.
  2. Group update energy by Flow component.
  3. Measure what fraction of total Full-Flow update energy lies inside the
     current 96 LoRA target weights (target coverage).
  4. Exact SVD of each target Delta-W matrix.
  5. Measure weighted rank-4/8/16/32/64 energy capture.
  6. Compute total-flow representable fraction = target coverage * rank capture.

Observed r=8 LoRA single-task SR reference:
  T6=0.70, T7=0.98, T8=0.86, T9=0.34
EOF

echo "==========================================================" | tee "${LOG_FILE}"
echo " LaWAM Goal Full-Flow Update + Rank Analysis" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"
echo "Analysis GPU : ${ANALYSIS_GPU}" | tee -a "${LOG_FILE}"
echo "Output       : ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "SVD ranks    : 4 8 16 32 64" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

python "${ANALYSIS_SCRIPT}" \
    --base "${BASE_CKPT}" \
    --task T6 6 "${T6_CKPT}" \
    --task T7 7 "${T7_CKPT}" \
    --task T8 8 "${T8_CKPT}" \
    --task T9 9 "${T9_CKPT}" \
    --ranks 4 8 16 32 64 \
    --device cuda:0 \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

echo
echo "=========================================================="
echo " Analysis complete"
echo "=========================================================="
echo "Main summary:"
echo "  ${OUTPUT_DIR}/SUMMARY.txt"
echo "Task overview:"
echo "  ${OUTPUT_DIR}/task_overview.csv"
echo "Component localization:"
echo "  ${OUTPUT_DIR}/component_update_summary.csv"
echo "SVD weighted summary:"
echo "  ${OUTPUT_DIR}/svd_weighted_summary.csv"
echo "Per-target SVD:"
echo "  ${OUTPUT_DIR}/svd_per_target_tensor.csv"
echo "Per-tensor update:"
echo "  ${OUTPUT_DIR}/per_tensor_update.csv"
echo "LoRA SR reference:"
echo "  ${OUTPUT_DIR}/lora_sr_reference.csv"
echo "=========================================================="