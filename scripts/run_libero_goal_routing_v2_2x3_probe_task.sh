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
[ -f "${SKILL_CKPT}" ] || { echo "[ERROR] missing GT skill ckpt ${SKILL_CKPT}"; exit 1; }

VARIANTS=(taskwm_hdhz taskwm_hdh taskwm_dh basewm_hdhz basewm_hdh basewm_dh)
for t in 6 7 8 9; do
  [ -f "${V2_ROOT}/task${t}/routing_memory/routing_memory.pt" ] || { echo "[ERROR] missing T${t} original routing_memory.pt"; exit 1; }
  [ -f "${V2_ROOT}/task${t}/routing_upstream_delta/routing_upstream_delta.pt" ] || { echo "[ERROR] missing T${t} routing_upstream_delta.pt"; exit 1; }
  for v in "${VARIANTS[@]}"; do
    p="${V2_ROOT}/task${t}/routing_memory_variants/${v}/dynamics_ae.pt"
    [ -f "${p}" ] || { echo "[ERROR] missing T${t}/${v}: ${p}"; exit 1; }
  done
done

NUM_TRIALS="${NUM_TRIALS:-2}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DEBUG_DECISIONS="${DEBUG_DECISIONS:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
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
reserve_port "${STAR_VLA_PYTHON}" "${PORT_BASE:-6894}" "${PORT_SEARCH_LIMIT:-200}"
PORT="${RESERVED_PORT}"
SERVER_PID=""
cleanup(){
  local code=$?
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  release_reserved_port
  exit "${code}"
}
trap cleanup EXIT INT TERM

PROBE_JSONL="${TASK_OUT}/dynamics_2x3_probe.jsonl"
SERVER_LOG="${SUITE_DIR}/server.log"
EVAL_LOG="${SUITE_DIR}/eval.log"
rm -f "${PROBE_JSONL}"

SEM_ARGS=()
DELTA_ARGS=()
DYN_ARGS=()
for t in 6 7 8 9; do
  SEM_ARGS+=(--semantic-memory "${t}=${V2_ROOT}/task${t}/routing_memory/routing_memory.pt")
  DELTA_ARGS+=(--upstream-delta "${t}=${V2_ROOT}/task${t}/routing_upstream_delta/routing_upstream_delta.pt")
  for v in "${VARIANTS[@]}"; do
    DYN_ARGS+=(--dynamics-memory "${v}:${t}=${V2_ROOT}/task${t}/routing_memory_variants/${v}/dynamics_ae.pt")
  done
done

echo "======================================================================"
echo " Routing-V2 PAIRED 2x3 passive Dynamics probe — T${TASK_ID}"
echo " Robot execution : oracle/task-ID Skill T${TASK_ID}"
echo " Semantic        : one shared Base-anchor Semantic Top-2"
echo " Candidate future: Task-WM + Shared Base-WM, computed on SAME chunks"
echo " Variants        : ${VARIANTS[*]}"
echo " Routing control : DISABLED; none of the six variants controls robot"
echo " Trials          : ${NUM_TRIALS}"
echo " Output          : ${TASK_OUT}"
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${POLICY_GPU}" "${STAR_VLA_PYTHON}" \
  deployment/model_server/server_policy_routing_v2_2x3_probe.py \
  --ckpt_path "${SKILL_CKPT}" \
  --port "${PORT}" --use_bf16 \
  --gt-task-id "${TASK_ID}" \
  --probe-output "${PROBE_JSONL}" \
  --debug-decisions "${DEBUG_DECISIONS}" \
  "${SEM_ARGS[@]}" "${DELTA_ARGS[@]}" "${DYN_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Wait for the server's own listening log instead of opening a raw TCP probe.
# This avoids the misleading "invalid websocket handshake" warning.
STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT_SEC:-1200}"
SERVER_READY=false
for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] 2x3 probe server exited during startup"
    tail -n 160 "${SERVER_LOG}" || true
    exit 1
  fi
  if grep -q "Websocket policy server listening" "${SERVER_LOG}" 2>/dev/null; then
    SERVER_READY=true
    break
  fi
  sleep 1
done
if [ "${SERVER_READY}" != true ]; then
  echo "[ERROR] 2x3 probe server startup timed out after ${STARTUP_TIMEOUT}s"
  tail -n 160 "${SERVER_LOG}" || true
  exit 1
fi
echo "[OK] Routing-V2 2x3 probe server ready on port ${PORT}"

EVAL_CMD=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py
  --args.pretrained-path "${SKILL_CKPT}"
  --args.host 127.0.0.1 --args.port "${PORT}"
  --args.task-suite-name libero_goal
  --args.num-trials-per-task "${NUM_TRIALS}"
  --args.num-workers "${EVAL_WORKERS}"
  --args.worker-sync-timeout-sec "${WORKER_SYNC_TIMEOUT_SEC:-1.0}"
  --args.worker-result-timeout-sec "${WORKER_RESULT_TIMEOUT_SEC:-1800}"
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
  echo "[ERROR] rollout failed ${STATUS}"
  tail -n 160 "${SERVER_LOG}" || true
  exit "${STATUS}"
fi

python scripts/summarize_libero_cl_eval.py \
  --run-dir "${SUITE_DIR}" --task-ids "${TASK_ID}" --expected-trials "${NUM_TRIALS}"
[ -s "${PROBE_JSONL}" ] || { echo "[ERROR] no 2x3 probe decisions"; exit 1; }
echo "[OK] T${TASK_ID} paired 2x3 probe complete decisions=$(wc -l < "${PROBE_JSONL}")"
