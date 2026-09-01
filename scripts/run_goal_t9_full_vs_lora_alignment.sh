#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
ANALYSIS_SCRIPT="${ROOT}/scripts/analyze_flow_full_vs_lora_alignment.py"

BASE_CKPT="${BASE_CKPT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft/20260807_124027+base_t0_5_10k_4gpu_bs32_ga2/final_model/pytorch_model.pt}"
FULL_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/full_flow_t9_retrain"
FULL_POINTER="${FULL_ROOT}/LATEST_T9_FULL_FLOW.txt"

if [ -n "${FULL_T9_CKPT:-}" ]; then :
elif [ -f "${FULL_POINTER}" ]; then
    FULL_T9_CKPT=$(head -n 1 "${FULL_POINTER}")
else
    FULL_RUN=$(find "${FULL_ROOT}/heads" -maxdepth 1 -type d -name '*+t9_*step_*gpu_bs*_ga*_full_flow_only_from_base' | sort | tail -n 1)
    [ -n "${FULL_RUN}" ] || { echo "[ERROR] Full-Flow T9 checkpoint not found. Run the retrain script first."; exit 1; }
    FULL_T9_CKPT="${FULL_RUN}/final_model/pytorch_model.pt"
fi

if [ -z "${LORA_R8_T9_CKPT:-}" ]; then
    R8_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_lora_oracle/r8/heads"
    R8_RUN=$(find "${R8_ROOT}" -maxdepth 1 -type d -name '*+cl4_t9_2000step_4gpu_bs64_ga1_flow_lora_r8_from_base' | sort | tail -n 1)
    [ -n "${R8_RUN}" ] || { echo "[ERROR] r8 T9 run not found"; exit 1; }
    LORA_R8_T9_CKPT="${R8_RUN}/final_model/pytorch_model_merged.pt"
fi

if [ -z "${LORA_R32_T9_CKPT:-}" ]; then
    R32_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_lora_oracle/r32_a32_lr0p0001/heads"
    R32_RUN=$(find "${R32_ROOT}" -maxdepth 1 -type d -name '*+cl4_t9_2000step_4gpu_bs64_ga1_flow_lora_r32_a32_lr0p0001_from_base' | sort | tail -n 1)
    [ -n "${R32_RUN}" ] || { echo "[ERROR] r32 T9 run not found under ${R32_ROOT}"; exit 1; }
    LORA_R32_T9_CKPT="${R32_RUN}/final_model/pytorch_model_merged.pt"
fi

for f in "${ANALYSIS_SCRIPT}" "${BASE_CKPT}" "${FULL_T9_CKPT}" "${LORA_R8_T9_CKPT}" "${LORA_R32_T9_CKPT}"; do
    [ -f "${f}" ] || { echo "[ERROR] Missing required file: ${f}"; exit 1; }
done

ANALYSIS_GPU="${ANALYSIS_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ROOT}/results/analysis/lawam_cl/libero_goal/t9_full_vs_lora_alignment/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/CHECKPOINTS.txt" <<EOT
Base:
${BASE_CKPT}

Strict Frozen-Upstream Full-Flow T9:
${FULL_T9_CKPT}

Flow-LoRA r8/a8/lr1e-4 T9 (SR=0.34):
${LORA_R8_T9_CKPT}

Flow-LoRA r32/a32/lr1e-4 T9 (SR=0.48):
${LORA_R32_T9_CKPT}
EOT

echo "=========================================================="
echo " T9 Strict Full-Flow vs Flow-LoRA Update Alignment"
echo "=========================================================="
echo "Analysis GPU : ${ANALYSIS_GPU}"
echo "Base         : ${BASE_CKPT}"
echo "Full T9      : ${FULL_T9_CKPT}"
echo "LoRA r8      : ${LORA_R8_T9_CKPT}"
echo "LoRA r32     : ${LORA_R32_T9_CKPT}"
echo "Output       : ${OUTPUT_DIR}"
echo "=========================================================="

python "${ANALYSIS_SCRIPT}" \
    --base "${BASE_CKPT}" \
    --full "${FULL_T9_CKPT}" \
    --candidate r8_a8_lr1e-4 "${LORA_R8_T9_CKPT}" \
    --candidate r32_a32_lr1e-4 "${LORA_R32_T9_CKPT}" \
    --device cuda:0 \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/analysis.log"

echo "=========================================================="
echo " Alignment analysis complete"
echo "=========================================================="
echo "Main summary : ${OUTPUT_DIR}/SUMMARY.txt"
echo "Overview     : ${OUTPUT_DIR}/alignment_overview.csv"
echo "Components   : ${OUTPUT_DIR}/component_alignment.csv"
echo "Blocks       : ${OUTPUT_DIR}/block_alignment.csv"
echo "Per tensor   : ${OUTPUT_DIR}/per_tensor_alignment.csv"
echo "=========================================================="