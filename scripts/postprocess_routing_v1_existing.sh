#!/usr/bin/env bash
set -euo pipefail

# Rebuild manifest/SR matrix/CL metrics/routing diagnostics from an already
# completed Routing-V1 rollout directory. No simulator or model inference is run.
# Usage:
#   PROTOCOL=cl_only bash scripts/postprocess_routing_v1_existing.sh /path/to/EVAL_STAMP_DIR
#   PROTOCOL=base_inclusive bash scripts/postprocess_routing_v1_existing.sh /path/to/EVAL_STAMP_DIR

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

OUT_ROOT="${1:-${EXISTING_OUT_ROOT:-}}"
[ -n "${OUT_ROOT}" ] || { echo "Usage: PROTOCOL=cl_only|base_inclusive $0 /path/to/existing/run"; exit 2; }
OUT_ROOT=$(realpath "${OUT_ROOT}")
[ -d "${OUT_ROOT}" ] || { echo "[ERROR] Existing run directory not found: ${OUT_ROOT}"; exit 1; }
PROTOCOL="${PROTOCOL:-}"
if [ -z "${PROTOCOL}" ] && [ -f "${OUT_ROOT}/PROTOCOL.txt" ]; then
  PROTOCOL=$(awk -F= '$1=="PROTOCOL"{print $2}' "${OUT_ROOT}/PROTOCOL.txt" | tail -n1)
fi
case "${PROTOCOL}" in cl_only|base_inclusive) ;; *) echo "[ERROR] Set PROTOCOL=cl_only or base_inclusive"; exit 2 ;; esac

MANIFEST="${OUT_ROOT}/manifest.csv"
ROUTING_LOG_INDEX="${OUT_ROOT}/routing_log_paths.txt"
"${STAR_VLA_PYTHON}" - "${MANIFEST}" <<'PY'
import csv, sys
with open(sys.argv[1], "w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow(["stage", "candidates", "task_ids", "summary_csv", "routing_summary", "routing_log"])
PY
: > "${ROUTING_LOG_INDEX}"

append_manifest_row() {
  "${STAR_VLA_PYTHON}" - "${MANIFEST}" "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import csv, sys
with open(sys.argv[1], "a", encoding="utf-8", newline="") as f:
    csv.writer(f).writerow(sys.argv[2:])
PY
}

find_suite_dir() {
  local stage="$1"
  find "${OUT_ROOT}/${stage}" -type d -path '*/suites/libero_goal' 2>/dev/null | sort | tail -n 1
}

record_stage() {
  local stage="$1" candidates="$2" task_ids="$3" competition="$4"
  local suite_dir
  suite_dir=$(find_suite_dir "${stage}")
  [ -n "${suite_dir}" ] || { echo "[ERROR] Could not find suites/libero_goal under ${OUT_ROOT}/${stage}"; exit 1; }
  local summary_csv="${suite_dir}/per_task_summary.csv"
  [ -f "${summary_csv}" ] || { echo "[ERROR] Missing ${summary_csv}"; exit 1; }
  if [ "${competition}" = "yes" ]; then
    local routing_summary="${suite_dir}/routing_summary/routing_summary.json"
    local routing_log="${suite_dir}/routing_chunks.jsonl"
    [ -f "${routing_summary}" ] || { echo "[ERROR] Missing ${routing_summary}"; exit 1; }
    [ -s "${routing_log}" ] || { echo "[ERROR] Missing/empty ${routing_log}"; exit 1; }
    append_manifest_row "${stage}" "${candidates}" "${task_ids}" "${summary_csv}" "${routing_summary}" "${routing_log}"
    echo "${routing_log}" >> "${ROUTING_LOG_INDEX}"
  else
    append_manifest_row "${stage}" "${candidates}" "${task_ids}" "${summary_csv}" "" ""
  fi
  echo "[POST][CHECK] ${stage}: ${summary_csv}"
}

if [ "${PROTOCOL}" = "cl_only" ]; then
  record_stage CL1 "t6" "6" yes
  record_stage CL2 "t6,t7" "6 7" yes
  record_stage CL3 "t6,t7,t8" "6 7 8" yes
  record_stage CL4 "t6,t7,t8,t9" "6 7 8 9" yes
  TASK_ARGS=(6 7 8 9)
  MATRIX="${OUT_ROOT}/sr_matrix_cl_only.csv"
else
  record_stage Base "base" "0 1 2 3 4 5" no
  record_stage CL1 "base,t6" "0 1 2 3 4 5 6" yes
  record_stage CL2 "base,t6,t7" "0 1 2 3 4 5 6 7" yes
  record_stage CL3 "base,t6,t7,t8" "0 1 2 3 4 5 6 7 8" yes
  record_stage CL4 "base,t6,t7,t8,t9" "0 1 2 3 4 5 6 7 8 9" yes
  TASK_ARGS=(0 1 2 3 4 5 6 7 8 9)
  MATRIX="${OUT_ROOT}/sr_matrix_base_inclusive.csv"
fi

"${STAR_VLA_PYTHON}" scripts/build_routing_v1_sr_matrix.py \
  --manifest "${MANIFEST}" --tasks "${TASK_ARGS[@]}" --protocol "${PROTOCOL}" --output "${MATRIX}"
"${STAR_VLA_PYTHON}" scripts/compute_routing_v1_metrics.py \
  --matrix "${MATRIX}" --protocol "${PROTOCOL}" --output-dir "${OUT_ROOT}/metrics"

ALL_ROUTING_LOG="${OUT_ROOT}/routing_chunks_all_stages.jsonl"
: > "${ALL_ROUTING_LOG}"
while IFS= read -r p; do
  [ -s "${p}" ] || { echo "[ERROR] Missing routing log ${p}"; exit 1; }
  cat "${p}" >> "${ALL_ROUTING_LOG}"
done < "${ROUTING_LOG_INDEX}"
"${STAR_VLA_PYTHON}" scripts/summarize_routing_v1_logs.py \
  --log "${ALL_ROUTING_LOG}" --output-dir "${OUT_ROOT}/routing_summary_all_stages"

echo "======================================================================"
echo " Routing-V1 postprocess complete (NO rollout was rerun)"
echo " Protocol : ${PROTOCOL}"
echo " Root     : ${OUT_ROOT}"
echo " Matrix   : ${MATRIX}"
echo " Metrics  : ${OUT_ROOT}/metrics/metrics.json"
echo " Routing  : ${OUT_ROOT}/routing_summary_all_stages/routing_summary.json"
echo "======================================================================"
