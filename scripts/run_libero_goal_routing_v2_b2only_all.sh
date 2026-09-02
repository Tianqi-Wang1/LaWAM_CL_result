#!/usr/bin/env bash
set -euo pipefail

# Formal order:
#   1) Base T0-T5 (10k, full LaWAM benchmark post-training)
#   2) independent T6-T9 skills (10k each, all fresh from the same Base)
#   3) optional task-ID oracle evaluation
#   4) per-task Semantic + Dynamics AEs (one 5k trajectory, retain 1k/2k/5k)

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2_b2only}"
BASE_STEPS="${BASE_STEPS:-10000}"
SKILL_STEPS="${SKILL_STEPS:-10000}"
MEMORY_STEPS="${MEMORY_STEPS:-5000}"
MEMORY_SNAPSHOT_STEPS="${MEMORY_SNAPSHOT_STEPS:-1000,2000,5000}"
NUM_TRIALS="${NUM_TRIALS:-50}"

SKIP_BASE="${SKIP_BASE:-false}"
SKIP_SKILLS="${SKIP_SKILLS:-false}"
RUN_TASKID_EVAL="${RUN_TASKID_EVAL:-true}"
SKIP_MEMORIES="${SKIP_MEMORIES:-false}"

export ROOT V2_ROOT
mkdir -p "${V2_ROOT}"

echo "======================================================================"
echo " Routing-V2 B2-only full experiment"
echo " Root          : ${V2_ROOT}"
echo " Base          : ${BASE_STEPS} steps"
echo " Skills T6-T9 : ${SKILL_STEPS} steps/task, independent from same Base"
echo " Skill path    : VLM Text-LoRA + Action-B2 only"
echo " AE training   : ${MEMORY_STEPS} steps; keep ${MEMORY_SNAPSHOT_STEPS}"
echo " Task-ID eval  : ${RUN_TASKID_EVAL} (${NUM_TRIALS} trials/task)"
echo "======================================================================"

if [ "${SKIP_BASE}" = "true" ]; then
  [ -f "${V2_ROOT}/latest_base_run.txt" ] || {
    echo "[ERROR] SKIP_BASE=true but latest_base_run.txt is missing"; exit 1;
  }
  BASE_RUN=$(cat "${V2_ROOT}/latest_base_run.txt")
  [ -f "${BASE_RUN}/final_model/pytorch_model.pt" ] || {
    echo "[ERROR] Existing Base final checkpoint is missing"; exit 1;
  }
  echo "[INFO] Reusing Base: ${BASE_RUN}"
else
  MAX_TRAIN_STEPS="${BASE_STEPS}" \
  NUM_WARMUP_STEPS="${BASE_WARMUP_STEPS:-600}" \
  SAVE_INTERVAL="$((BASE_STEPS + 1))" \
    bash "${ROOT}/scripts/run_libero_goal_routing_v2_base.sh"
fi

if [ "${SKIP_SKILLS}" = "true" ]; then
  for task in 6 7 8 9; do
    pointer="${V2_ROOT}/task${task}/latest_skill_run.txt"
    [ -f "${pointer}" ] || { echo "[ERROR] Missing ${pointer}"; exit 1; }
    run=$(cat "${pointer}")
    [ -f "${run}/final_model/pytorch_model.pt" ] || {
      echo "[ERROR] Missing existing T${task} skill final checkpoint"; exit 1;
    }
  done
  echo "[INFO] Reusing all existing T6-T9 skill checkpoints."
else
  for task in 6 7 8 9; do
    TASK_ID="${task}" \
    MAX_TRAIN_STEPS="${SKILL_STEPS}" \
    NUM_WARMUP_STEPS="${SKILL_WARMUP_STEPS:-600}" \
      bash "${ROOT}/scripts/run_libero_goal_routing_v2_b2only_skill_task.sh"
  done
fi

if [ "${RUN_TASKID_EVAL}" = "true" ]; then
  NUM_TRIALS="${NUM_TRIALS}" \
    bash "${ROOT}/scripts/eval_libero_goal_routing_v2_taskid_all.sh"
fi

if [ "${SKIP_MEMORIES}" = "true" ]; then
  echo "[INFO] SKIP_MEMORIES=true; stopping after skill training/evaluation."
  exit 0
fi

for task in 6 7 8 9; do
  TASK_ID="${task}" \
  MAX_TRAIN_STEPS="${MEMORY_STEPS}" \
  NUM_WARMUP_STEPS="${MEMORY_WARMUP_STEPS:-250}" \
  MEMORY_SNAPSHOT_STEPS="${MEMORY_SNAPSHOT_STEPS}" \
  SAVE_INTERVAL="${MEMORY_SAVE_INTERVAL:-1000}" \
    bash "${ROOT}/scripts/run_libero_goal_routing_v2_b2only_memory_task.sh"
done

echo "======================================================================"
echo " Routing-V2 B2-only training complete"
echo " Base/skill policy storage : final_model/pytorch_model.pt only"
echo " AE storage                : step_1000, step_2000, step_5000"
echo " Root                      : ${V2_ROOT}"
echo "======================================================================"
