#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# LaWAM World-Model / Dynamics Forgetting
# Formal 2x2 module intervention:
#
#                    Base LaWM W_B       CL LaWM W_k
#   teacher z*       W_B(h_t,z*)         W_k(h_t,z*)      <- wm_only
#   CL z_k           W_B(h_t,z_k)        W_k(h_t,z_k)     <- full
#                    ^ upstream_only
#
# Formal observation protocol:
#   Base tasks 0-5
#   all filtered demonstration trajectories
#   25% / 50% / 75% anchors
#   correct old-task instruction
#
# No MuJoCo / online LIBERO evaluation is required.
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
SKIP_TEACHER_STABILITY_CHECK="${SKIP_TEACHER_STABILITY_CHECK:-False}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/analysis/lawam_cl/world_model_intervention}"
SCRIPT="${ROOT}/scripts/analyze_libero_world_forgetting_v3.py"

test -f "${SCRIPT}" || { echo "[ERROR] missing ${SCRIPT}"; exit 1; }

if [ "${SPLIT}" != "all" ]; then
  echo "[WARN] Formal protocol uses SPLIT=all; current=${SPLIT}"
fi
if [ "${MAX_TRAJECTORIES_PER_TASK}" -gt 0 ]; then
  echo "[WARN] MAX_TRAJECTORIES_PER_TASK=${MAX_TRAJECTORIES_PER_TASK}: smoke/debug mode"
fi

reuse=()
case "$(echo "${REUSE_INPUT_CACHE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) reuse=(--reuse-input-cache) ;;
  0|false|no|n|off) ;;
  *) echo "[ERROR] REUSE_INPUT_CACHE must be True/False"; exit 1 ;;
esac

teacher_flag=()
case "$(echo "${SKIP_TEACHER_STABILITY_CHECK}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on) teacher_flag=(--skip-teacher-stability-check) ;;
  0|false|no|n|off) ;;
  *) echo "[ERROR] SKIP_TEACHER_STABILITY_CHECK must be True/False"; exit 1 ;;
esac

echo "=========================================================="
echo "LaWAM World-Model / Dynamics Forgetting"
echo "2x2 Module Intervention"
echo "=========================================================="
echo "GPU                    : ${GPU_ID}"
echo "Suites                 : ${SUITES}"
echo "Base tasks             : ${TASK_IDS}"
echo "Trajectory fractions   : ${TRAJECTORY_FRACTIONS}"
echo "Max trajectories/task  : ${MAX_TRAJECTORIES_PER_TASK} (0 = all)"
echo "Batch size             : ${BATCH_SIZE}"
echo "Workers                : ${NUM_WORKERS}"
echo "Split                  : ${SPLIT}"
echo "Reuse input cache      : ${REUSE_INPUT_CACHE}"
echo "Skip teacher check     : ${SKIP_TEACHER_STABILITY_CHECK}"
echo "Output root            : ${OUTPUT_ROOT}"
echo "Run tag                : ${TIMESTAMP}"
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
  echo "run root : ${RUN_ROOT}"
  echo "output   : ${OUT}"
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
    "${reuse[@]}" \
    "${teacher_flag[@]}"
done

echo
echo "=========================================================="
echo "Goal/Object world-model intervention analysis completed"
echo "Output root: ${OUTPUT_ROOT}"
echo "Run tag    : ${TIMESTAMP}"
echo "=========================================================="