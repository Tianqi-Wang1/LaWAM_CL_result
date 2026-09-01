#!/usr/bin/env bash
set -euo pipefail

# Routing-V1 task-agnostic closed-loop evaluation.
# Two protocols are intentionally separate evaluations:
#   cl_only        : only T6..T9 experts compete; Base never enters the bank.
#   base_inclusive : Base competes with T6..T9 experts as the bank expands.
# GT task IDs are diagnostics-only metadata and are stripped before model inference.

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
ROUTING_ROOT="${ROUTING_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_base10k_expert2k}"
MODE="${MODE:-b1}"
PROTOCOL="${PROTOCOL:-cl_only}"
SCORE_MODE="${SCORE_MODE:-world}"
ALPHA_INPUT="${ALPHA:-}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS:-5}"
ROUTING_EXECUTION_MODE="${ROUTING_EXECUTION_MODE:-route}"
SCORE_NORMALIZATION="${SCORE_NORMALIZATION:-none}"
TEMPORAL_MODE="${TEMPORAL_MODE:-none}"
TEMPORAL_BETA="${TEMPORAL_BETA:-0.0}"
TEMPORAL_MARGIN="${TEMPORAL_MARGIN:-0.0}"
STAGES="${STAGES:-all}"
PROTOCOL_POSTPROCESS="${PROTOCOL_POSTPROCESS:-auto}"
BASE_SUMMARY_CSV_OVERRIDE="${BASE_SUMMARY_CSV_OVERRIDE:-}"
EVAL_STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}"

case "${MODE}" in b1|b2) ;; *) echo "[ERROR] MODE must be b1 or b2"; exit 2 ;; esac
case "${PROTOCOL}" in cl_only|base_inclusive) ;; *) echo "[ERROR] PROTOCOL must be cl_only or base_inclusive"; exit 2 ;; esac
case "${SCORE_MODE}" in latent|world|combined) ;; *) echo "[ERROR] SCORE_MODE must be latent/world/combined"; exit 2 ;; esac
case "${ROUTING_EXECUTION_MODE}" in route|oracle_execute) ;; *) echo "[ERROR] ROUTING_EXECUTION_MODE must be route/oracle_execute"; exit 2 ;; esac
case "${SCORE_NORMALIZATION}" in none|candidate_mean) ;; *) echo "[ERROR] SCORE_NORMALIZATION must be none/candidate_mean"; exit 2 ;; esac
case "${TEMPORAL_MODE}" in none|ema|ema_hysteresis) ;; *) echo "[ERROR] TEMPORAL_MODE must be none/ema/ema_hysteresis"; exit 2 ;; esac

# Canonicalize alpha so directory names/logs cannot silently claim a different
# weighting than the requested score mode.
case "${SCORE_MODE}" in
  world) ALPHA="0.0" ;;
  latent) ALPHA="1.0" ;;
  combined) ALPHA="${ALPHA_INPUT:-0.5}" ;;
esac

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

BASE_RUN="${BASE_RUN:-}"
if [ -z "${BASE_RUN}" ] && [ -f "${ROUTING_ROOT}/latest_base_run.txt" ]; then BASE_RUN=$(cat "${ROUTING_ROOT}/latest_base_run.txt"); fi
if [ -z "${BASE_RUN}" ]; then BASE_RUN=$(find "${ROUTING_ROOT}/base_runs" -maxdepth 1 -type d -name '*+routing_v1_base_t0_5_*' 2>/dev/null | sort | tail -n 1); fi
[ -n "${BASE_RUN}" ] || { echo "[ERROR] Routing-V1 Base run not found"; exit 1; }
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
[ -f "${BASE_CKPT}" ] || { echo "[ERROR] Missing Base checkpoint: ${BASE_CKPT}"; exit 1; }

find_latest_run() {
  local mode="$1" task="$2"
  find "${ROUTING_ROOT}/${mode}/task${task}/runs" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}
for task in 6 7 8 9; do
  run=$(find_latest_run "${MODE}" "${task}")
  [ -n "${run}" ] || { echo "[ERROR] Missing ${MODE^^} T${task} run"; exit 1; }
  ckpt="${run}/final_model/pytorch_model.pt"
  [ -f "${ckpt}" ] || { echo "[ERROR] Missing ${ckpt}"; exit 1; }
  eval "EXPERT_${task}_CKPT=\"${ckpt}\""
done

ROUTING_VARIANT="${SCORE_MODE}_a${ALPHA}_norm-${SCORE_NORMALIZATION}_temp-${TEMPORAL_MODE}_b${TEMPORAL_BETA}_m${TEMPORAL_MARGIN}"
OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_no_taskid/${MODE}/${PROTOCOL}/${ROUTING_VARIANT}/${EVAL_STAMP}"
mkdir -p "${OUT_ROOT}"
MANIFEST="${OUT_ROOT}/manifest.csv"
"${STAR_VLA_PYTHON}" - "${MANIFEST}" <<'PY'
import csv, sys
with open(sys.argv[1], "w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow(["stage", "candidates", "task_ids", "summary_csv", "routing_summary", "routing_log"])
PY
ROUTING_LOG_INDEX="${OUT_ROOT}/routing_log_paths.txt"
: > "${ROUTING_LOG_INDEX}"

append_manifest_row() {
  local stage="$1" candidates="$2" task_ids="$3" summary_csv="$4" routing_summary="$5" routing_log="$6"
  "${STAR_VLA_PYTHON}" - "${MANIFEST}" "${stage}" "${candidates}" "${task_ids}" "${summary_csv}" "${routing_summary}" "${routing_log}" <<'PY'
import csv, sys
path, stage, candidates, task_ids, summary_csv, routing_summary, routing_log = sys.argv[1:]
with open(path, "a", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow([stage, candidates, task_ids, summary_csv, routing_summary, routing_log])
PY
}

stage_enabled() {
  local wanted="$1"
  if [ "${STAGES}" = "all" ]; then
    return 0
  fi
  for x in ${STAGES}; do
    if [ "${x}" = "${wanted}" ]; then return 0; fi
  done
  return 1
}

run_routing_stage() {
  local stage="$1" candidates="$2" task_ids="$3"
  local stage_out="${OUT_ROOT}/${stage}"
  local alias="routing_${MODE}_${PROTOCOL}_${SCORE_MODE}_${stage}"
  mkdir -p "${stage_out}"
  echo "======================================================================"
  echo " Routing-V1 stage=${stage} mode=${MODE} protocol=${PROTOCOL} score=${SCORE_MODE} alpha=${ALPHA}"
  echo " candidates : ${candidates}"
  echo " eval tasks  : ${task_ids}"
  if [ "${ROUTING_EXECUTION_MODE}" = "oracle_execute" ]; then
    echo " GT task ID  : stripped before model scoring; used only AFTER scoring to execute the diagnostic oracle expert"
  else
    echo " GT task ID  : diagnostics only; stripped before policy batch construction"
  fi
  echo " execution   : ${ROUTING_EXECUTION_MODE}"
  echo " score norm  : ${SCORE_NORMALIZATION}"
  echo " temporal    : ${TEMPORAL_MODE} beta=${TEMPORAL_BETA} margin=${TEMPORAL_MARGIN}"
  echo " cadence     : one routing decision per action chunk (temporal state only smooths scores)"
  echo " common noise: one flow-noise tensor shared by all candidates per chunk"
  if [ "${MODE}" = "b2" ]; then
    echo " B2 upstream : each candidate activates its own Text-LoRA before z_k/h_k self-consistency"
  else
    echo " B1 upstream : VLM/QFormer/LaWM reference z/h shared once across candidates"
  fi
  echo "======================================================================"

  ROUTING_BASE_CKPT="${BASE_CKPT}" \
  ROUTING_EXPERT_6_CKPT="${EXPERT_6_CKPT}" \
  ROUTING_EXPERT_7_CKPT="${EXPERT_7_CKPT}" \
  ROUTING_EXPERT_8_CKPT="${EXPERT_8_CKPT}" \
  ROUTING_EXPERT_9_CKPT="${EXPERT_9_CKPT}" \
  ROUTING_MODE="${MODE}" ROUTING_SCORE_MODE="${SCORE_MODE}" ROUTING_ALPHA="${ALPHA}" \
  ROUTING_SCORE_NORMALIZATION="${SCORE_NORMALIZATION}" \
  ROUTING_TEMPORAL_MODE="${TEMPORAL_MODE}" ROUTING_TEMPORAL_BETA="${TEMPORAL_BETA}" ROUTING_TEMPORAL_MARGIN="${TEMPORAL_MARGIN}" \
  ROUTING_EXECUTION_MODE="${ROUTING_EXECUTION_MODE}" \
  ROUTING_CANDIDATES="${candidates}" ROUTING_CONTEXT_LABEL="${stage}" ROUTING_DEBUG_REQUESTS="${ROUTING_DEBUG_REQUESTS}" \
  TASK_IDS="${task_ids}" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" NUM_WORKERS="${EVAL_WORKERS}" \
  GPU_ID="${POLICY_GPU}" EVAL_GPU_ID="${EVAL_GPU}" SAVE_VIDEOS="${SAVE_VIDEOS}" \
  OUTPUT_ROOT="${stage_out}" LIBERO_CKPT_ALIAS="${alias}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_routing_v1_suite_benchmark.sh \
    "${EXPERT_6_CKPT}" libero_goal 0

  local suite_dir
  suite_dir=$(find "${stage_out}/${alias}" -type d -path '*/suites/libero_goal' | sort | tail -n 1)
  [ -n "${suite_dir}" ] || { echo "[ERROR] suite dir not found for ${stage}"; exit 1; }
  "${STAR_VLA_PYTHON}" scripts/summarize_libero_cl_eval.py --run-dir "${suite_dir}" --task-ids ${task_ids} --expected-trials "${NUM_TRIALS}"
  local routing_summary="${suite_dir}/routing_summary/routing_summary.json"
  local routing_log="${suite_dir}/routing_chunks.jsonl"
  local summary_csv="${suite_dir}/per_task_summary.csv"
  [ -f "${summary_csv}" ] || { echo "[ERROR] Per-task summary missing: ${summary_csv}"; exit 1; }
  [ -f "${routing_summary}" ] || { echo "[ERROR] Routing summary missing: ${routing_summary}"; exit 1; }
  [ -s "${routing_log}" ] || { echo "[ERROR] Routing chunk log missing/empty: ${routing_log}"; exit 1; }
  echo "${routing_log}" >> "${ROUTING_LOG_INDEX}"
  append_manifest_row "${stage}" "${candidates}" "${task_ids}" "${summary_csv}" "${routing_summary}" "${routing_log}"
  echo "[RoutingV1][CHECK] stage=${stage} artifacts recorded in manifest with CSV-safe quoting."
}

run_base_stage() {
  if [ -n "${BASE_SUMMARY_CSV_OVERRIDE}" ]; then
    [ -f "${BASE_SUMMARY_CSV_OVERRIDE}" ] || { echo "[ERROR] BASE_SUMMARY_CSV_OVERRIDE missing: ${BASE_SUMMARY_CSV_OVERRIDE}"; exit 1; }
    echo "[Routing-V1] Reusing shared Base acquisition summary (no Base rollout): ${BASE_SUMMARY_CSV_OVERRIDE}"
    append_manifest_row "Base" "base" "0 1 2 3 4 5" "${BASE_SUMMARY_CSV_OVERRIDE}" "" ""
    return
  fi

  local stage_out="${OUT_ROOT}/Base"
  local alias="routing_${MODE}_base_reference"
  mkdir -p "${stage_out}"
  echo "[Routing-V1] Base acquisition row: Base is the only available expert (no competition)."
  TASK_IDS="0 1 2 3 4 5" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" NUM_WORKERS="${EVAL_WORKERS}" \
  GPU_ID="${POLICY_GPU}" EVAL_GPU_ID="${EVAL_GPU}" SAVE_VIDEOS="${SAVE_VIDEOS}" \
  OUTPUT_ROOT="${stage_out}" LIBERO_CKPT_ALIAS="${alias}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_suite_benchmark.sh "${BASE_CKPT}" libero_goal 0
  local suite_dir
  suite_dir=$(find "${stage_out}/${alias}" -type d -path '*/suites/libero_goal' | sort | tail -n 1)
  [ -n "${suite_dir}" ] || { echo "[ERROR] Base suite dir not found"; exit 1; }
  "${STAR_VLA_PYTHON}" scripts/summarize_libero_cl_eval.py --run-dir "${suite_dir}" --task-ids 0 1 2 3 4 5 --expected-trials "${NUM_TRIALS}"
  append_manifest_row "Base" "base" "0 1 2 3 4 5" "${suite_dir}/per_task_summary.csv" "" ""
}

postprocess_current_run() {
  if [ "${PROTOCOL}" = "cl_only" ]; then
    MATRIX="${OUT_ROOT}/sr_matrix_cl_only.csv"
    TASK_ARGS=(6 7 8 9)
  else
    MATRIX="${OUT_ROOT}/sr_matrix_base_inclusive.csv"
    TASK_ARGS=(0 1 2 3 4 5 6 7 8 9)
  fi

  "${STAR_VLA_PYTHON}" scripts/build_routing_v1_sr_matrix.py \
    --manifest "${MANIFEST}" --tasks "${TASK_ARGS[@]}" --protocol "${PROTOCOL}" --output "${MATRIX}"
  "${STAR_VLA_PYTHON}" scripts/compute_routing_v1_metrics.py \
    --matrix "${MATRIX}" --protocol "${PROTOCOL}" --output-dir "${OUT_ROOT}/metrics"

  # Aggregate routing-stage chunk logs. The Base-only acquisition row has no
  # routing competition and is intentionally absent.
  ALL_ROUTING_LOG="${OUT_ROOT}/routing_chunks_all_stages.jsonl"
  : > "${ALL_ROUTING_LOG}"
  while IFS= read -r routing_log_path; do
    [ -s "${routing_log_path}" ] || { echo "[ERROR] Missing routing log listed in ${ROUTING_LOG_INDEX}: ${routing_log_path}"; exit 1; }
    cat "${routing_log_path}" >> "${ALL_ROUTING_LOG}"
  done < "${ROUTING_LOG_INDEX}"
  if [ -s "${ALL_ROUTING_LOG}" ]; then
    "${STAR_VLA_PYTHON}" scripts/summarize_routing_v1_logs.py \
      --log "${ALL_ROUTING_LOG}" \
      --output-dir "${OUT_ROOT}/routing_summary_all_stages"
  else
    echo "[ERROR] No routing decisions were recorded for protocol ${PROTOCOL}." >&2
    exit 1
  fi
}

if [ "${PROTOCOL}" = "cl_only" ]; then
  stage_enabled CL1 && run_routing_stage CL1 "t6" "6"
  stage_enabled CL2 && run_routing_stage CL2 "t6,t7" "6 7"
  stage_enabled CL3 && run_routing_stage CL3 "t6,t7,t8" "6 7 8"
  stage_enabled CL4 && run_routing_stage CL4 "t6,t7,t8,t9" "6 7 8 9"
else
  stage_enabled Base && run_base_stage
  stage_enabled CL1 && run_routing_stage CL1 "base,t6" "0 1 2 3 4 5 6"
  stage_enabled CL2 && run_routing_stage CL2 "base,t6,t7" "0 1 2 3 4 5 6 7"
  stage_enabled CL3 && run_routing_stage CL3 "base,t6,t7,t8" "0 1 2 3 4 5 6 7 8"
  stage_enabled CL4 && run_routing_stage CL4 "base,t6,t7,t8,t9" "0 1 2 3 4 5 6 7 8 9"
fi

DO_POSTPROCESS="${PROTOCOL_POSTPROCESS}"
if [ "${DO_POSTPROCESS}" = "auto" ]; then
  if [ "${STAGES}" = "all" ]; then DO_POSTPROCESS=true; else DO_POSTPROCESS=false; fi
fi
case "${DO_POSTPROCESS}" in true|false) ;; *) echo "[ERROR] PROTOCOL_POSTPROCESS must be auto/true/false"; exit 2 ;; esac

# Always aggregate whatever routing logs were produced. Full SR matrix/CL metrics
# require the complete canonical stage set and are skipped for targeted diagnostics.
ALL_ROUTING_LOG="${OUT_ROOT}/routing_chunks_all_stages.jsonl"
: > "${ALL_ROUTING_LOG}"
while IFS= read -r routing_log_path; do
  [ -s "${routing_log_path}" ] || { echo "[ERROR] Missing routing log listed in ${ROUTING_LOG_INDEX}: ${routing_log_path}"; exit 1; }
  cat "${routing_log_path}" >> "${ALL_ROUTING_LOG}"
done < "${ROUTING_LOG_INDEX}"
if [ -s "${ALL_ROUTING_LOG}" ]; then
  "${STAR_VLA_PYTHON}" scripts/summarize_routing_v1_logs.py \
    --log "${ALL_ROUTING_LOG}" \
    --output-dir "${OUT_ROOT}/routing_summary_all_stages"
else
  echo "[ERROR] No routing decisions were recorded for stages=${STAGES}." >&2
  exit 1
fi

MATRIX="not_built_targeted_diagnostic"
if [ "${DO_POSTPROCESS}" = "true" ]; then
  # postprocess_current_run also re-aggregates routing logs, which is harmless
  # and keeps the full-protocol behavior unchanged.
  postprocess_current_run
fi

cat > "${OUT_ROOT}/PROTOCOL.txt" <<EOF
Routing-V1 no-task-ID evaluation
MODE=${MODE}
PROTOCOL=${PROTOCOL}
SCORE_MODE=${SCORE_MODE}
ALPHA=${ALPHA}
SCORE_NORMALIZATION=${SCORE_NORMALIZATION}
TEMPORAL_MODE=${TEMPORAL_MODE}
TEMPORAL_BETA=${TEMPORAL_BETA}
TEMPORAL_MARGIN=${TEMPORAL_MARGIN}
NUM_TRIALS=${NUM_TRIALS}
STAGES=${STAGES}
ROUTING_EXECUTION_MODE=${ROUTING_EXECUTION_MODE}
Base=${BASE_CKPT}
T6=${EXPERT_6_CKPT}
T7=${EXPERT_7_CKPT}
T8=${EXPERT_8_CKPT}
T9=${EXPERT_9_CKPT}
Important: routing_gt_task_id is always stripped before LaWAM batch construction. In route mode it is diagnostics-only; in oracle_execute diagnostics it is consulted only AFTER all candidate scores are computed to choose the executed expert.
EOF

echo "======================================================================"
echo " Routing-V1 evaluation complete"
echo " Output   : ${OUT_ROOT}"
echo " Manifest : ${MANIFEST}"
echo " Matrix   : ${MATRIX}"
if [ "${DO_POSTPROCESS}" = "true" ]; then
  echo " Metrics  : ${OUT_ROOT}/metrics/metrics.json"
else
  echo " Metrics  : skipped (targeted stages=${STAGES})"
fi
echo " Routing  : ${OUT_ROOT}/routing_summary_all_stages/routing_summary.json"
echo " Context  : ${OUT_ROOT}/routing_summary_all_stages/routing_per_context.csv"
echo " CtxTask  : ${OUT_ROOT}/routing_summary_all_stages/routing_per_context_task.csv"
echo "======================================================================"
