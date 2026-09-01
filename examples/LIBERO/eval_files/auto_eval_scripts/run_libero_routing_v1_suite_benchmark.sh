#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(libero_repo_root)"
cd "${REPO_ROOT}"

your_ckpt="${1:-${CKPT_PATH:-}}"
task_suite_name="${2:-${TASK_SUITE_NAME:-}}"
run_index="${3:-${RUN_INDEX:-}}"

if [ -z "${your_ckpt}" ] || [ -z "${task_suite_name}" ] || [ -z "${run_index}" ]; then
  echo "Usage: $0 <ckpt_path> <task_suite_name> <run_index>" >&2
  exit 1
fi
if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi

# Routing-V1 bank settings. `your_ckpt` is the template checkpoint used by the
# ordinary LIBERO client to read config/statistics and must match server metadata.
routing_base_ckpt="${ROUTING_BASE_CKPT:-}"
routing_mode="${ROUTING_MODE:-b1}"
routing_score_mode="${ROUTING_SCORE_MODE:-world}"
routing_alpha="${ROUTING_ALPHA:-0.5}"
routing_score_normalization="${ROUTING_SCORE_NORMALIZATION:-none}"
routing_temporal_mode="${ROUTING_TEMPORAL_MODE:-none}"
routing_temporal_beta="${ROUTING_TEMPORAL_BETA:-0.0}"
routing_temporal_margin="${ROUTING_TEMPORAL_MARGIN:-0.0}"
routing_execution_mode="${ROUTING_EXECUTION_MODE:-route}"
routing_candidates="${ROUTING_CANDIDATES:-}"
routing_debug_requests="${ROUTING_DEBUG_REQUESTS:-5}"
routing_context_label="${ROUTING_CONTEXT_LABEL:-}"
if [ -z "${routing_base_ckpt}" ] || [ ! -f "${routing_base_ckpt}" ]; then
  echo "[ERROR] ROUTING_BASE_CKPT must point to the Routing-V1 Base checkpoint." >&2
  exit 1
fi
if [ -z "${routing_candidates}" ]; then
  echo "[ERROR] ROUTING_CANDIDATES is required, e.g. t6,t7 or base,t6,t7." >&2
  exit 1
fi
case "${routing_mode}" in b1|b2) ;; *) echo "[ERROR] ROUTING_MODE must be b1 or b2" >&2; exit 1 ;; esac
case "${routing_score_mode}" in latent|world|combined) ;; *) echo "[ERROR] ROUTING_SCORE_MODE must be latent/world/combined" >&2; exit 1 ;; esac
case "${routing_execution_mode}" in route|oracle_execute) ;; *) echo "[ERROR] ROUTING_EXECUTION_MODE must be route/oracle_execute" >&2; exit 1 ;; esac
case "${routing_score_normalization}" in none|candidate_mean) ;; *) echo "[ERROR] ROUTING_SCORE_NORMALIZATION must be none/candidate_mean" >&2; exit 1 ;; esac
case "${routing_temporal_mode}" in none|ema|ema_hysteresis) ;; *) echo "[ERROR] ROUTING_TEMPORAL_MODE must be none/ema/ema_hysteresis" >&2; exit 1 ;; esac

export LIBERO_HOME="${LIBERO_HOME:-../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_python:-python}}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${starVLA_python:-/usr/local/miniconda3/bin/python}}"
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"

if [ ! -x "${LIBERO_PYTHON}" ]; then
  echo "[ERROR] LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi
if [ ! -x "${STAR_VLA_PYTHON}" ]; then
  echo "[ERROR] STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 1
fi
if [ ! -d "${LIBERO_HOME}" ]; then
  echo "[ERROR] LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  exit 1
fi

ensure_libero_eval_config "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}"

num_gpus="$(detect_num_gpus "${STAR_VLA_PYTHON}")"
if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]] || [ "${num_gpus}" -lt 1 ]; then
  echo "[ERROR] NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
  exit 1
fi

gpu_id="${GPU_ID:-}"
if [ -z "${gpu_id}" ]; then
  gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${GPU_IDS:-}")"
fi
eval_gpu_id="${EVAL_GPU_ID:-}"
if [ -z "${eval_gpu_id}" ] && [ -n "${EVAL_GPU_IDS:-}" ]; then
  eval_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${EVAL_GPU_IDS}")"
fi
if [ -z "${eval_gpu_id}" ]; then
  eval_gpu_id="${gpu_id}"
fi
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
num_workers="${NUM_WORKERS:-1}"
if [ -z "${MUJOCO_GL:-}" ]; then
  if [ "${num_workers}" -gt 1 ]; then
    export MUJOCO_GL="osmesa"
  else
    export MUJOCO_GL="egl"
  fi
fi
if [ -z "${PYOPENGL_PLATFORM:-}" ] && { [ "${MUJOCO_GL}" = "egl" ] || [ "${MUJOCO_GL}" = "osmesa" ]; }; then
  export PYOPENGL_PLATFORM="${MUJOCO_GL}"
fi
max_tasks="${MAX_TASKS:-}"
task_ids="${TASK_IDS:-}"
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
save_similarity_video="${SAVE_SIMILARITY_VIDEO:-False}"
sim_src_row="${SIM_SRC_ROW:-3}"
sim_src_col="${SIM_SRC_COL:-7}"
sim_vmin="${SIM_VMIN:-0.4}"
sim_vmax="${SIM_VMAX:-1.0}"
sim_alpha="${SIM_ALPHA:-0.5}"
sim_cmap="${SIM_CMAP:-jet}"
host="${HOST:-127.0.0.1}"
port_base="${PORT_BASE:-5694}"
preferred_port=$((port_base + run_index))
port_search_limit="${PORT_SEARCH_LIMIT:-200}"
reserve_port "${STAR_VLA_PYTHON}" "${preferred_port}" "${port_search_limit}"
base_port="${RESERVED_PORT}"
server_startup_timeout_sec="${SERVER_STARTUP_TIMEOUT_SEC:-600}"
benchmark_variant="${BENCHMARK_VARIANT:-libero}"
enable_category_aggregation="${ENABLE_CATEGORY_AGGREGATION:-False}"
unnorm_key="${UNNORM_KEY:-}"
log_path="${LOG_PATH:-}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
worker_sync_timeout_sec="${WORKER_SYNC_TIMEOUT_SEC:-1.0}"
eval_action_chunk_len="${EVAL_ACTION_CHUNK_LEN:-}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"

output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero}"
run_group="${LIBERO_RUN_GROUP:-${LIBERO_CKPT_ALIAS:-$(derive_ckpt_alias "${your_ckpt}")}}"
run_tag="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
suite_dir="${output_root}/${run_group}/${run_tag}/suites/${task_suite_name}"
mkdir -p "${suite_dir}"

server_log="${suite_dir}/server.log"
eval_log="${suite_dir}/eval.log"
routing_log="${suite_dir}/routing_chunks.jsonl"
server_pid=""

eval_cmd=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py
  --args.pretrained-path "${your_ckpt}"
  --args.host "${host}"
  --args.port "${base_port}"
  --args.task-suite-name "${task_suite_name}"
  --args.num-trials-per-task "${num_trials_per_task}"
  --args.num-workers "${num_workers}"
  --args.worker-sync-timeout-sec "${worker_sync_timeout_sec}"
  --args.video-out-path "${suite_dir}"
  --args.benchmark-variant "${benchmark_variant}"
  --args.enable-category-aggregation "${enable_category_aggregation}"
  --args.worker-result-timeout-sec "${worker_result_timeout_sec}"
)

if [ "${save_videos}" = "True" ]; then
  eval_cmd+=(--args.save-videos)
else
  eval_cmd+=(--args.no-save-videos)
fi
if [ "${save_only_failure_videos}" = "True" ]; then
  eval_cmd+=(--args.save-only-failure-videos)
else
  eval_cmd+=(--args.no-save-only-failure-videos)
fi
if [ "${save_similarity_video}" = "True" ]; then
  eval_cmd+=(--args.save-similarity-video)
else
  eval_cmd+=(--args.no-save-similarity-video)
fi
eval_cmd+=(
  --args.sim-src-row "${sim_src_row}"
  --args.sim-src-col "${sim_src_col}"
  --args.sim-vmin "${sim_vmin}"
  --args.sim-vmax "${sim_vmax}"
  --args.sim-alpha "${sim_alpha}"
  --args.sim-cmap "${sim_cmap}"
)

if [ -n "${unnorm_key}" ]; then
  eval_cmd+=(--args.unnorm-key "${unnorm_key}")
fi
if [ -n "${max_tasks}" ]; then
  eval_cmd+=(--args.max-tasks "${max_tasks}")
fi
if [ -n "${task_ids}" ]; then

  if [ -n "${max_tasks}" ]; then
    echo \
      "[ERROR] TASK_IDS and MAX_TASKS cannot be used together." \
      >&2
    exit 1
  fi

  # Accept either:
  #   TASK_IDS="0 1 2"
  # or
  #   TASK_IDS="0,1,2"
  normalized_task_ids="$(
    printf '%s' "${task_ids}" \
    | tr ',' ' '
  )"

  read -r -a TASK_ID_ARRAY \
    <<< "${normalized_task_ids}"

  eval_cmd+=(
    --args.task-ids
    "${TASK_ID_ARRAY[@]}"
  )
fi
if [ -n "${log_path}" ]; then
  eval_cmd+=(--args.log-path "${log_path}")
fi
if [ -n "${eval_action_chunk_len}" ]; then
  eval_cmd+=(--args.eval-action-chunk-len "${eval_action_chunk_len}")
fi

cleanup() {
  local exit_code=$?
  if [ -n "${server_pid}" ] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  release_reserved_port
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "Starting LIBERO suite benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  task_suite: ${task_suite_name}"
echo "  run_index: ${run_index}"
echo "  server_gpu_id: ${gpu_id}"
echo "  eval_gpu_id: ${eval_gpu_id}"
echo "  num_workers: ${num_workers}"
echo "  max_tasks: ${max_tasks:-all}"
echo "  task_ids: ${task_ids:-all}"
echo "  mujoco_gl: ${MUJOCO_GL}"
echo "  save_videos: ${save_videos}"
echo "  save_only_failure_videos: ${save_only_failure_videos}"
echo "  save_similarity_video: ${save_similarity_video}"
echo "  sim_src_row: ${sim_src_row}"
echo "  sim_src_col: ${sim_src_col}"
echo "  sim_vmin: ${sim_vmin}"
echo "  sim_vmax: ${sim_vmax}"
echo "  sim_alpha: ${sim_alpha}"
echo "  sim_cmap: ${sim_cmap}"
echo "  worker_sync_timeout_sec: ${worker_sync_timeout_sec}"
echo "  worker_result_timeout_sec: ${worker_result_timeout_sec}"
echo "  server_startup_timeout_sec: ${server_startup_timeout_sec}"
echo "  eval_action_chunk_len: ${eval_action_chunk_len:-full_chunk}"
echo "  port: ${base_port}"
if [ "${save_only_failure_videos}" = "True" ] && [ "${save_videos}" != "True" ]; then
  echo "[WARN] SAVE_ONLY_FAILURE_VIDEOS=True is ignored because SAVE_VIDEOS=False." >&2
  echo "[WARN] Set SAVE_VIDEOS=True SAVE_ONLY_FAILURE_VIDEOS=True to save only failed episodes." >&2
fi
if [ "${base_port}" != "${preferred_port}" ]; then
  echo "  preferred_port: ${preferred_port} (occupied, auto-switched)"
fi
echo "  output: ${suite_dir}"
echo "  routing_mode: ${routing_mode}"
echo "  routing_score_mode: ${routing_score_mode}"
echo "  routing_alpha: ${routing_alpha}"
echo "  routing_score_normalization: ${routing_score_normalization}"
echo "  routing_temporal_mode: ${routing_temporal_mode}"
echo "  routing_temporal_beta: ${routing_temporal_beta}"
echo "  routing_temporal_margin: ${routing_temporal_margin}"
echo "  routing_execution_mode: ${routing_execution_mode}"
echo "  routing_candidates: ${routing_candidates}"
echo "  routing_base_ckpt: ${routing_base_ckpt}"
echo "  routing_context: ${routing_context_label:-none}"

routing_server_cmd=(
  "${STAR_VLA_PYTHON}" deployment/model_server/server_policy_routing_v1.py
  --base_ckpt "${routing_base_ckpt}"
  --candidates "${routing_candidates}"
  --mode "${routing_mode}"
  --score_mode "${routing_score_mode}"
  --alpha "${routing_alpha}"
  --score_normalization "${routing_score_normalization}"
  --temporal_mode "${routing_temporal_mode}"
  --temporal_beta "${routing_temporal_beta}"
  --temporal_margin "${routing_temporal_margin}"
  --execution_mode "${routing_execution_mode}"
  --routing_log "${routing_log}"
  --context_label "${routing_context_label}"
  --require_gt_diagnostics
  --debug_requests "${routing_debug_requests}"
  --port "${base_port}"
  --use_bf16
)
for task in 6 7 8 9; do
  var="ROUTING_EXPERT_${task}_CKPT"
  ckpt="${!var:-}"
  if [ -n "${ckpt}" ]; then
    [ -f "${ckpt}" ] || { echo "[ERROR] ${var} not found: ${ckpt}" >&2; exit 1; }
    routing_server_cmd+=(--expert "${task}=${ckpt}")
  fi
done

CUDA_VISIBLE_DEVICES="${gpu_id}" "${routing_server_cmd[@]}" > "${server_log}" 2>&1 &
server_pid=$!

if ! wait_for_port "${STAR_VLA_PYTHON}" "${host}" "${base_port}" "${server_startup_timeout_sec}"; then
  echo "[ERROR] Policy server failed to become ready on ${host}:${base_port}" >&2
  echo "[ERROR] server_log: ${server_log}" >&2
  echo "[ERROR] Try increasing SERVER_STARTUP_TIMEOUT_SEC if the checkpoint is still loading." >&2
  tail -n 40 "${server_log}" >&2 || true
  exit 1
fi

set +e
PYTHONFAULTHANDLER=1 \
CUDA_VISIBLE_DEVICES="${eval_gpu_id}" MUJOCO_EGL_DEVICE_ID="${eval_gpu_id}" \
  "${eval_cmd[@]}" 2>&1 | tee "${eval_log}"
eval_status=${PIPESTATUS[0]}
set -e

if [ "${eval_status}" -ne 0 ]; then
  echo "[ERROR] Evaluation command failed with exit code ${eval_status}" >&2
  tail -n 40 "${server_log}" >&2 || true
  exit "${eval_status}"
fi

if [ -f "${routing_log}" ]; then
  "${STAR_VLA_PYTHON}" scripts/summarize_routing_v1_logs.py \
    --log "${routing_log}" --output-dir "${suite_dir}/routing_summary"
else
  echo "[WARN] Routing log was not created: ${routing_log}" >&2
fi

echo "LIBERO Routing-V1 suite benchmark completed: ${suite_dir}"
