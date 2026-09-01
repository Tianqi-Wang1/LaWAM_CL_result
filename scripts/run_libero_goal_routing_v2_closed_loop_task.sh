#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"

STAGE="${STAGE:?Set STAGE=CL1/CL2/CL3/CL4}"
TASK_ID="${TASK_ID:?Set TASK_ID=6/7/8/9}"
CANDIDATE_TASKS="${CANDIDATE_TASKS:?Set CANDIDATE_TASKS, e.g. '6 7'}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DEBUG_DECISIONS="${DEBUG_DECISIONS:-8}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.20}"
LAMBDA_MAX="${LAMBDA_MAX:-0.50}"
FUSION_GAMMA="${FUSION_GAMMA:-2.0}"

case " ${CANDIDATE_TASKS} " in *" ${TASK_ID} "*) ;; *) echo "[ERROR] GT task T${TASK_ID} not in candidate bank: ${CANDIDATE_TASKS}"; exit 2;; esac

export LIBERO_HOME="${LIBERO_HOME:-/home/jincai_guo/tianqi/CVPR2027/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python}"
export PYTHONPATH="${LIBERO_HOME}:${ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

candidate_args=()
memory_args=()
TEMPLATE_CKPT=""
GT_CKPT=""
for t in ${CANDIDATE_TASKS}; do
  run_file="${V2_ROOT}/task${t}/latest_skill_run.txt"
  [ -f "${run_file}" ] || { echo "[ERROR] missing ${run_file}"; exit 1; }
  run=$(cat "${run_file}")
  ckpt="${run}/final_model/pytorch_model.pt"
  mem="${V2_ROOT}/task${t}/routing_memory/routing_memory.pt"
  [ -f "${ckpt}" ] || { echo "[ERROR] missing T${t} skill: ${ckpt}"; exit 1; }
  [ -f "${mem}" ] || { echo "[ERROR] missing T${t} routing memory: ${mem}"; exit 1; }
  candidate_args+=(--candidate-skill "${t}=${ckpt}")
  memory_args+=(--routing-memory "${t}=${mem}")
  if [ -z "${TEMPLATE_CKPT}" ]; then TEMPLATE_CKPT="${ckpt}"; fi
  if [ "${t}" = "${TASK_ID}" ]; then GT_CKPT="${ckpt}"; fi
done
[ -n "${GT_CKPT}" ] || { echo "[ERROR] could not resolve GT checkpoint T${TASK_ID}"; exit 1; }

# IMPORTANT: ModelClient uses --args.pretrained-path only to read local
# config/normalization and to validate the server metadata checkpoint identity.
# It does NOT execute that checkpoint; all actions are produced by the routing
# server.  The routing server is materialized from TEMPLATE_CKPT (the first
# candidate, T6 in CL-only), so the client must use the same logical config
# anchor.  All V2 skill checkpoints were already audited to share the same
# normalization/action configuration as the V2 Base.
CLIENT_CONFIG_CKPT="${TEMPLATE_CKPT}"

TASK_OUT="${OUTPUT_ROOT}/T${TASK_ID}"
SUITE_DIR="${TASK_OUT}/suites/libero_goal"
mkdir -p "${SUITE_DIR}"
ROUTING_LOG="${TASK_OUT}/routing_chunks.jsonl"
SERVER_LOG="${SUITE_DIR}/server.log"
EVAL_LOG="${SUITE_DIR}/eval.log"
rm -f "${ROUTING_LOG}"

source examples/LIBERO/eval_files/auto_eval_scripts/common.sh
ensure_libero_eval_config "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}"
reserve_port "${STAR_VLA_PYTHON}" "${PORT_BASE:-6094}" "${PORT_SEARCH_LIMIT:-300}"
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

cat > "${TASK_OUT}/PROTOCOL.txt" <<EOF
Routing-V2 CLOSED-LOOP / ${STAGE} / true LIBERO task T${TASK_ID}
Candidate bank: ${CANDIDATE_TASKS}
GT task ID: server-side diagnostics only; NEVER used by selection or policy input.
Selection per action chunk:
  Base anchor -> Semantic AE bank -> normalized-gap confidence.
  If C_sem >= ${GATE_THRESHOLD}: Semantic Top-1.
  If C_sem < ${GATE_THRESHOLD}: Semantic Top-2 WAM imagination + Dynamics AE.
  Pair-normalized fusion: S=(1-lambda(C))*Semantic + lambda(C)*Dynamics.
  lambda(C)=${LAMBDA_MAX}*((delta-C)/delta)^${FUSION_GAMMA}, delta=${GATE_THRESHOLD}.
Selected Skill Path controls the actual action chunk.
Eval-client config/metadata anchor: ${CLIENT_CONFIG_CKPT}
True-task skill checkpoint (diagnostics/bank member, NOT client config anchor): ${GT_CKPT}
EOF

echo "======================================================================"
echo " Routing-V2 CLOSED-LOOP ${STAGE} / evaluate true task T${TASK_ID}"
echo " Candidates       : ${CANDIDATE_TASKS}"
echo " Gate             : C_sem < ${GATE_THRESHOLD}"
echo " Adaptive fusion  : lambda_max=${LAMBDA_MAX}, gamma=${FUSION_GAMMA}"
echo " GT task ID       : DIAGNOSTICS ONLY"
echo " Server anchor    : ${TEMPLATE_CKPT}"
echo " Client cfg anchor: ${CLIENT_CONFIG_CKPT}"
echo " True-task skill  : ${GT_CKPT}"
echo " Trials           : ${NUM_TRIALS}"
echo " Output           : ${TASK_OUT}"
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${POLICY_GPU}" "${STAR_VLA_PYTHON}" \
  deployment/model_server/server_policy_routing_v2_closed_loop.py \
  --ckpt_path "${TEMPLATE_CKPT}" \
  --port "${PORT}" --use_bf16 \
  --gt-task-id "${TASK_ID}" --stage "${STAGE}" \
  --routing-log "${ROUTING_LOG}" \
  --gate-threshold "${GATE_THRESHOLD}" \
  --lambda-max "${LAMBDA_MAX}" \
  --gamma "${FUSION_GAMMA}" \
  --debug-decisions "${DEBUG_DECISIONS}" \
  "${candidate_args[@]}" "${memory_args[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Do not use common.sh::wait_for_port here: it opens a raw TCP connection
# which the WebSocket server logs as an invalid HTTP/WebSocket handshake.
# Instead wait for the server's own readiness log while also checking that the
# process is still alive.
SERVER_READY_TIMEOUT="${SERVER_STARTUP_TIMEOUT_SEC:-1200}"
SERVER_READY_START=$(date +%s)
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] Routing-V2 server exited before becoming ready"
    tail -n 200 "${SERVER_LOG}" || true
    exit 1
  fi
  if grep -q "Websocket policy server listening on ws://0.0.0.0:${PORT}" "${SERVER_LOG}" 2>/dev/null; then
    break
  fi
  now=$(date +%s)
  if [ $((now - SERVER_READY_START)) -ge "${SERVER_READY_TIMEOUT}" ]; then
    echo "[ERROR] Timed out waiting for Routing-V2 server on port ${PORT}"
    tail -n 200 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 1
done
echo "[OK] Routing-V2 server ready on port ${PORT}"

EVAL_CMD=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py
  --args.pretrained-path "${CLIENT_CONFIG_CKPT}"
  --args.host 127.0.0.1 --args.port "${PORT}"
  --args.task-suite-name libero_goal
  --args.num-trials-per-task "${NUM_TRIALS}"
  --args.num-workers "${EVAL_WORKERS}"
  --args.worker-sync-timeout-sec "${WORKER_SYNC_TIMEOUT_SEC:-1.0}"
  --args.worker-result-timeout-sec "${WORKER_RESULT_TIMEOUT_SEC:-1200}"
  --args.video-out-path "${SUITE_DIR}"
  --args.task-ids "${TASK_ID}"
  --args.no-save-similarity-video
)
if [ "${SAVE_VIDEOS}" = "True" ]; then EVAL_CMD+=(--args.save-videos); else EVAL_CMD+=(--args.no-save-videos); fi
EVAL_CMD+=(--args.no-save-only-failure-videos)

set +e
PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES="${EVAL_GPU}" MUJOCO_EGL_DEVICE_ID="${EVAL_GPU}" \
  "${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_LOG}"
STATUS=${PIPESTATUS[0]}
set -e
if [ "${STATUS}" -ne 0 ]; then
  echo "[ERROR] rollout failed with status ${STATUS}"
  tail -n 160 "${SERVER_LOG}" || true
  exit "${STATUS}"
fi

python scripts/summarize_libero_cl_eval.py \
  --run-dir "${SUITE_DIR}" --task-ids "${TASK_ID}" --expected-trials "${NUM_TRIALS}"
[ -s "${ROUTING_LOG}" ] || { echo "[ERROR] no routing decisions recorded"; exit 1; }

echo "[OK] ${STAGE}/T${TASK_ID} complete"
echo "     SR      : ${SUITE_DIR}/per_task_summary.csv"
echo "     Routing : ${ROUTING_LOG}"
