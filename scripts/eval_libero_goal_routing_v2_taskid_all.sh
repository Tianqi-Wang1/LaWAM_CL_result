#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"

V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
LATEST_BASE_FILE="${V2_ROOT}/latest_base_run.txt"
[ -f "${LATEST_BASE_FILE}" ] || { echo "[ERROR] Missing ${LATEST_BASE_FILE}"; exit 1; }
BASE_RUN="$(cat "${LATEST_BASE_FILE}")"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
[ -f "${BASE_CKPT}" ] || { echo "[ERROR] Missing V2 Base checkpoint: ${BASE_CKPT}"; exit 1; }

NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

STAMP="${EVAL_STAMP:-$(date +"%Y%m%d_%H%M%S")}" 
OUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_taskid/${STAMP}"
mkdir -p "${OUT_ROOT}"
MASTER_LOG="${OUT_ROOT}/taskid_eval.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "${OUT_ROOT}" > "${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v2_taskid/latest_eval_run.txt"

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

LAST_SUMMARY=""

run_one() {
  local ckpt="$1"
  local tasks="$2"
  local alias="$3"
  local out="$4"

  [ -f "${ckpt}" ] || { echo "[ERROR] Missing checkpoint for ${alias}: ${ckpt}"; exit 1; }
  mkdir -p "${out}"

  echo
  echo "======================================================================"
  echo " Routing-V2 task-ID evaluation"
  echo " Alias      : ${alias}"
  echo " Tasks      : ${tasks}"
  echo " Checkpoint : ${ckpt}"
  echo " Trials/task: ${NUM_TRIALS}"
  echo " Workers    : ${EVAL_WORKERS}"
  echo " Policy GPU : ${POLICY_GPU}"
  echo " Eval GPU   : ${EVAL_GPU}"
  echo " Output     : ${out}"
  echo "======================================================================"

  # IMPORTANT: do not wrap this call in command substitution.  Its stdout/stderr
  # remains attached to the terminal (and taskid_eval.log), so long LIBERO runs
  # show live progress instead of looking hung.
  SUITES="libero_goal" \
  TASK_IDS="${tasks}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
  NUM_WORKERS="${EVAL_WORKERS}" \
  GPU_IDS="${POLICY_GPU}" \
  EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" \
  OUTPUT_ROOT="${out}" \
  LIBERO_CKPT_ALIAS="${alias}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${ckpt}"

  local eval_dir suite_dir
  eval_dir="$(find "${out}/${alias}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  [ -n "${eval_dir}" ] || { echo "[ERROR] No evaluation directory found for ${alias} under ${out}/${alias}"; exit 1; }
  suite_dir="${eval_dir}/suites/libero_goal"
  [ -d "${suite_dir}" ] || { echo "[ERROR] Missing suite directory: ${suite_dir}"; exit 1; }

  "${STAR_VLA_PYTHON}" scripts/summarize_libero_cl_eval.py \
    --run-dir "${suite_dir}" \
    --task-ids ${tasks} \
    --expected-trials "${NUM_TRIALS}"

  LAST_SUMMARY="${suite_dir}/per_task_summary.csv"
  [ -f "${LAST_SUMMARY}" ] || { echo "[ERROR] Missing summary after ${alias}: ${LAST_SUMMARY}"; exit 1; }

  echo "[OK] ${alias} summary: ${LAST_SUMMARY}"
  cat "${LAST_SUMMARY}"
}

MANIFEST="${OUT_ROOT}/summaries.tsv"
printf 'source\tsummary_csv\n' > "${MANIFEST}"

run_one "${BASE_CKPT}" "0 1 2 3 4 5" "routing_v2_base" "${OUT_ROOT}/Base"
printf 'Base\t%s\n' "${LAST_SUMMARY}" >> "${MANIFEST}"

for t in 6 7 8 9; do
  latest_skill="${V2_ROOT}/task${t}/latest_skill_run.txt"
  [ -f "${latest_skill}" ] || { echo "[ERROR] Missing ${latest_skill}"; exit 1; }
  run="$(cat "${latest_skill}")"
  ckpt="${run}/final_model/pytorch_model.pt"

  run_one "${ckpt}" "${t}" "routing_v2_t${t}_taskid" "${OUT_ROOT}/T${t}"
  printf 'T%s\t%s\n' "${t}" "${LAST_SUMMARY}" >> "${MANIFEST}"
done

"${STAR_VLA_PYTHON}" - "${MANIFEST}" "${OUT_ROOT}/taskid_sr_summary.csv" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
with manifest.open("r", encoding="utf-8", newline="") as f:
    for item in csv.DictReader(f, delimiter="\t"):
        label = item["source"]
        path = Path(item["summary_csv"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing per-task summary for {label}: {path}")
        with path.open("r", encoding="utf-8", newline="") as sf:
            for r in csv.DictReader(sf):
                rows.append({"source": label, **r})

if not rows:
    raise RuntimeError("No task-ID evaluation rows were collected")

with out.open("w", encoding="utf-8", newline="") as f:
    fields = list(rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("[OK] wrote", out)
PY

cat > "${OUT_ROOT}/PROTOCOL.txt" <<EOF2
Routing-V2 task-ID sanity/formal evaluation.
No automatic routing is used.
Base T0-T5 uses the latest clean V2 Base checkpoint.
T6-T9 each use their own independently trained V2 skill checkpoint.
All evaluations are deferred until Base and all four skills/memories finish training.
Evaluation stdout/stderr is streamed live and saved to taskid_eval.log.
EOF2

echo
echo "======================================================================"
echo " Routing-V2 task-ID evaluation COMPLETE"
echo " Output  : ${OUT_ROOT}"
echo " Log     : ${MASTER_LOG}"
echo " Summary : ${OUT_ROOT}/taskid_sr_summary.csv"
echo "======================================================================"
cat "${OUT_ROOT}/taskid_sr_summary.csv"
