#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# LaWAM H_act Semantic Forgetting: Instruction Contrast
#
# Observation protocol:
#   Base tasks 0-5
#   every demonstration trajectory
#   25% / 50% / 75% progress anchors
#
# Instruction protocol for every fixed Base observation:
#   positive  = its own Base-task instruction
#   negative  = task 6 / 7 / 8 / 9 instructions
#
# Feature:
#   H_act only (8 latent-action-query hidden states)
#
# No MuJoCo / LIBERO online simulator is required.
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
BASE_TASK_IDS="${BASE_TASK_IDS:-0 1 2 3 4 5}"
NEGATIVE_TASK_IDS="${NEGATIVE_TASK_IDS:-6 7 8 9}"
TRAJECTORY_FRACTIONS="${TRAJECTORY_FRACTIONS:-0.25 0.50 0.75}"
MAX_TRAJECTORIES_PER_TASK="${MAX_TRAJECTORIES_PER_TASK:-0}"
ANCHORS_PER_BATCH="${ANCHORS_PER_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT="${SPLIT:-all}"
REUSE_ANCHOR_CACHE="${REUSE_ANCHOR_CACHE:-False}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/analysis/lawam_cl/semantic_instruction_contrast_hact}"
SCRIPT="${ROOT}/scripts/analyze_libero_instruction_contrast_hact.py"

if [ ! -f "${SCRIPT}" ]; then
  echo "[ERROR] missing ${SCRIPT}"
  exit 1
fi

if [ "${SPLIT}" != "all" ]; then
  echo "[WARN] Formal protocol uses SPLIT=all; current SPLIT=${SPLIT}"
fi
if [ "${MAX_TRAJECTORIES_PER_TASK}" -gt 0 ]; then
  echo "[WARN] MAX_TRAJECTORIES_PER_TASK=${MAX_TRAJECTORIES_PER_TASK}: smoke/debug mode."
fi

reuse=()
case "$(echo "${REUSE_ANCHOR_CACHE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) reuse=(--reuse-anchor-cache) ;;
  0|false|no|n|off) ;;
  *) echo "[ERROR] REUSE_ANCHOR_CACHE must be True/False"; exit 1 ;;
esac

NEG_COUNT=$(wc -w <<< "${NEGATIVE_TASK_IDS}")
ACTUAL_VLM_BATCH=$(( ANCHORS_PER_BATCH * (1 + NEG_COUNT) ))

echo "=========================================================="
echo "LaWAM H_act Semantic Forgetting: Instruction Contrast"
echo "=========================================================="
echo "GPU                     : ${GPU_ID}"
echo "Suites                  : ${SUITES}"
echo "Base observation tasks  : ${BASE_TASK_IDS}"
echo "Negative instruction IDs: ${NEGATIVE_TASK_IDS}"
echo "Trajectory fractions    : ${TRAJECTORY_FRACTIONS}"
echo "Max trajectories/task   : ${MAX_TRAJECTORIES_PER_TASK} (0 = all)"
echo "Anchors / raw batch     : ${ANCHORS_PER_BATCH}"
echo "Actual VLM batch size   : ${ACTUAL_VLM_BATCH}"
echo "Workers                 : ${NUM_WORKERS}"
echo "Split                   : ${SPLIT}"
echo "Reuse raw anchor cache  : ${REUSE_ANCHOR_CACHE}"
echo "Canonical config        : ${CONFIG_YAML}"
echo "Run tag                 : ${TIMESTAMP}"
echo "Output root             : ${OUTPUT_ROOT}"
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
    --base-task-ids ${BASE_TASK_IDS} \
    --negative-task-ids ${NEGATIVE_TASK_IDS} \
    --trajectory-fractions ${TRAJECTORY_FRACTIONS} \
    --max-trajectories-per-task "${MAX_TRAJECTORIES_PER_TASK}" \
    --anchors-per-batch "${ANCHORS_PER_BATCH}" \
    --num-workers "${NUM_WORKERS}" \
    --split "${SPLIT}" \
    --device cuda:0 \
    "${reuse[@]}"
done

echo
echo "=========================================================="
echo "Goal/Object H_act instruction-contrast analysis completed"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run tag    : ${TIMESTAMP}"
echo "=========================================================="