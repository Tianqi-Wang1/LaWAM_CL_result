#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# LaWAM Semantic / Conditioning Forgetting
# Formal trajectory-balanced protocol:
#   - Base tasks 0-5
#   - every filtered demonstration trajectory
#   - anchors at 25%, 50%, 75% trajectory progress
#   - fixed processed VLM inputs reused for Base / CL1-CL4
#   - no MuJoCo / LIBERO simulator required
# ==========================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

CONFIG_YAML="${CONFIG_YAML:-${ROOT}/starVLA/config/training/train_libero.yaml}"

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset NCCL_TOPO_FILE || true
unset NCCL_GRAPH_FILE || true
unset NCCL_CONF_FILE || true
unset HFAI_NCCL_OPT_LEVEL || true

GPU_ID="${GPU_ID:-2}"
SUITES="${SUITES:-libero_goal libero_object}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5}"
TRAJECTORY_FRACTIONS="${TRAJECTORY_FRACTIONS:-0.25 0.50 0.75}"
MAX_TRAJECTORIES_PER_TASK="${MAX_TRAJECTORIES_PER_TASK:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT="${SPLIT:-all}"
REUSE_INPUT_CACHE="${REUSE_INPUT_CACHE:-False}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/analysis/lawam_cl/semantic_conditioning_trajectory}"
SCRIPT="${ROOT}/scripts/analyze_libero_semantic_conditioning.py"

if [ ! -f "${SCRIPT}" ]; then
  echo "[ERROR] missing ${SCRIPT}"
  exit 1
fi

if [ "${SPLIT}" != "all" ]; then
  echo "[WARN] Formal protocol uses SPLIT=all; current SPLIT=${SPLIT}"
fi

if [ "${MAX_TRAJECTORIES_PER_TASK}" -gt 0 ]; then
  echo "[WARN] MAX_TRAJECTORIES_PER_TASK=${MAX_TRAJECTORIES_PER_TASK}: debug/smoke mode, not formal protocol."
fi

reuse=()
case "$(echo "${REUSE_INPUT_CACHE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) reuse=(--reuse-input-cache) ;;
  0|false|no|n|off) ;;
  *) echo "[ERROR] REUSE_INPUT_CACHE must be True/False"; exit 1 ;;
esac

echo "=========================================================="
echo "LaWAM Semantic / Conditioning Forgetting"
echo "Trajectory-balanced formal protocol"
echo "=========================================================="
echo "GPU                    : ${GPU_ID}"
echo "Suites                 : ${SUITES}"
echo "Base tasks             : ${TASK_IDS}"
echo "Split                  : ${SPLIT}"
echo "Trajectory fractions   : ${TRAJECTORY_FRACTIONS}"
echo "Max trajectories/task  : ${MAX_TRAJECTORIES_PER_TASK} (0 = all)"
echo "Canonical config       : ${CONFIG_YAML}"
echo "Batch size             : ${BATCH_SIZE}"
echo "Workers                : ${NUM_WORKERS}"
echo "Run tag                : ${TIMESTAMP}"
echo "Output root            : ${OUTPUT_ROOT}"
echo "=========================================================="

for SUITE in ${SUITES}; do
  case "${SUITE}" in
    libero_goal)
      RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
      ;;
    libero_object)
      RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"
      ;;
    *)
      echo "[ERROR] unsupported suite ${SUITE}"
      exit 1
      ;;
  esac

  OUT="${OUTPUT_ROOT}/${SUITE}/${TIMESTAMP}"
  mkdir -p "${OUT}"

  echo
  echo "----------------------------------------------------------"
  echo "[RUN] ${SUITE}"
  echo "      run root : ${RUN_ROOT}"
  echo "      output   : ${OUT}"
  echo "----------------------------------------------------------"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  python "${SCRIPT}" \
    --suite "${SUITE}" \
    --config-yaml "${CONFIG_YAML}" \
    --run-root "${RUN_ROOT}" \
    --output-dir "${OUT}" \
    --task-ids ${TASK_IDS} \
    --trajectory-fractions ${TRAJECTORY_FRACTIONS} \
    --max-trajectories-per-task "${MAX_TRAJECTORIES_PER_TASK}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --split "${SPLIT}" \
    --device cuda:0 \
    "${reuse[@]}"
done

echo
echo "=========================================================="
echo "Goal/Object trajectory-balanced analysis completed"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run tag    : ${TIMESTAMP}"
echo "=========================================================="