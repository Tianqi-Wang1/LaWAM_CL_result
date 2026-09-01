#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"
TASK_ID="${TASK_ID:?Set TASK_ID=6/7/8/9}"
case "${TASK_ID}" in 6|7|8|9) ;; *) echo "[ERROR] TASK_ID=${TASK_ID}"; exit 1;; esac

V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
SKILL_RUN="${SKILL_RUN:-$(cat "${V2_ROOT}/task${TASK_ID}/latest_skill_run.txt")}"
SKILL_CKPT="${SKILL_CKPT:-${SKILL_RUN}/final_model/pytorch_model.pt}"
[ -f "${SKILL_CKPT}" ] || { echo "[ERROR] Missing skill checkpoint: ${SKILL_CKPT}"; exit 1; }

for t in 6 7 8 9; do
  mem="${V2_ROOT}/task${t}/routing_memory/routing_memory.pt"
  [ -f "${mem}" ] || { echo "[ERROR] Missing T${t} routing memory: ${mem}"; exit 1; }
done

NUM_TRIALS="${NUM_TRIALS:-5}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DEBUG_DECISIONS="${DEBUG_DECISIONS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT to semantic-probe group directory}"
TASK_OUT="${OUTPUT_ROOT}/T${TASK_ID}"
SUITE_DIR="${TASK_OUT}/suites/libero_goal"
mkdir -p "${SUITE_DIR}"

export LIBERO_HOME="${LIBERO_HOME:-/home/jincai_guo/tianqi/CVPR2027/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python}"
export PYTHONPATH="${LIBERO_HOME}:${ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

source examples/LIBERO/eval_files/auto_eval_scripts/common.sh
ensure_libero_eval_config "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}"
reserve_port "${STAR_VLA_PYTHON}" "${PORT_BASE:-5794}" "${PORT_SEARCH_LIMIT:-200}"
PORT="${RESERVED_PORT}"
SERVER_PID=""
cleanup() {
  local code=$?
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  release_reserved_port
  exit "${code}"
}
trap cleanup EXIT INT TERM

PROBE_JSONL="${TASK_OUT}/semantic_probe.jsonl"
SERVER_LOG="${SUITE_DIR}/server.log"
EVAL_LOG="${SUITE_DIR}/eval.log"
rm -f "${PROBE_JSONL}"

MEM_ARGS=()
for t in 6 7 8 9; do
  MEM_ARGS+=(--semantic-memory "${t}=${V2_ROOT}/task${t}/routing_memory/routing_memory.pt")
done

echo "======================================================================"
echo " Routing-V2 Semantic AE passive probe — T${TASK_ID}"
echo " Execution       : task-ID/oracle Skill T${TASK_ID} (UNCHANGED)"
echo " Semantic bank   : T6 T7 T8 T9"
echo " Routing control : DISABLED; scores are diagnostics only"
echo " Metric          : per action-chunk Top-1 accuracy / Top-2 recall"
echo " Trials          : ${NUM_TRIALS}"
echo " Policy GPU      : ${POLICY_GPU}"
echo " Eval GPU        : ${EVAL_GPU}"
echo " Output          : ${TASK_OUT}"
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${POLICY_GPU}" "${STAR_VLA_PYTHON}" \
  deployment/model_server/server_policy_routing_v2_semantic_probe.py \
  --ckpt_path "${SKILL_CKPT}" \
  --port "${PORT}" \
  --use_bf16 \
  --gt-task-id "${TASK_ID}" \
  --probe-output "${PROBE_JSONL}" \
  --debug-decisions "${DEBUG_DECISIONS}" \
  "${MEM_ARGS[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

if ! wait_for_port "${STAR_VLA_PYTHON}" 127.0.0.1 "${PORT}" "${SERVER_STARTUP_TIMEOUT_SEC:-600}"; then
  echo "[ERROR] semantic-probe server failed to start" >&2
  tail -n 80 "${SERVER_LOG}" >&2 || true
  exit 1
fi

EVAL_CMD=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py
  --args.pretrained-path "${SKILL_CKPT}"
  --args.host 127.0.0.1
  --args.port "${PORT}"
  --args.task-suite-name libero_goal
  --args.num-trials-per-task "${NUM_TRIALS}"
  --args.num-workers "${EVAL_WORKERS}"
  --args.worker-sync-timeout-sec "${WORKER_SYNC_TIMEOUT_SEC:-1.0}"
  --args.worker-result-timeout-sec "${WORKER_RESULT_TIMEOUT_SEC:-600}"
  --args.video-out-path "${SUITE_DIR}"
  --args.task-ids "${TASK_ID}"
  --args.no-save-similarity-video
)
if [ "${SAVE_VIDEOS}" = "True" ]; then
  EVAL_CMD+=(--args.save-videos)
else
  EVAL_CMD+=(--args.no-save-videos)
fi
EVAL_CMD+=(--args.no-save-only-failure-videos)

set +e
PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES="${EVAL_GPU}" MUJOCO_EGL_DEVICE_ID="${EVAL_GPU}" \
  "${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_LOG}"
STATUS=${PIPESTATUS[0]}
set -e
if [ "${STATUS}" -ne 0 ]; then
  echo "[ERROR] LIBERO probe rollout failed: ${STATUS}" >&2
  tail -n 100 "${SERVER_LOG}" >&2 || true
  exit "${STATUS}"
fi

python scripts/summarize_libero_cl_eval.py \
  --run-dir "${SUITE_DIR}" \
  --task-ids "${TASK_ID}" \
  --expected-trials "${NUM_TRIALS}"

[ -s "${PROBE_JSONL}" ] || { echo "[ERROR] Semantic probe produced no decisions: ${PROBE_JSONL}"; exit 1; }
DECISIONS=$(wc -l < "${PROBE_JSONL}")
echo "[OK] T${TASK_ID} semantic probe complete: decisions=${DECISIONS} log=${PROBE_JSONL}"
