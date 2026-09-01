#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

ANALYSIS_SCRIPT="${ROOT}/scripts/analyze_lawam_full_flow_update_rank.py"

SEQFT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

BASE_CKPT="${SEQFT_ROOT}/20260807_124027+base_t0_5_10k_4gpu_bs32_ga2/final_model/pytorch_model.pt"
CL1_CKPT="${SEQFT_ROOT}/20260808_101711+cl1_t6_2k_4gpu_bs32_ga2/final_model/pytorch_model.pt"
CL2_CKPT="${SEQFT_ROOT}/20260808_111257+cl2_t7_2k_4gpu_bs32_ga2/final_model/pytorch_model.pt"
CL3_CKPT="${SEQFT_ROOT}/20260808_120836+cl3_t8_2k_4gpu_bs32_ga2/final_model/pytorch_model.pt"
CL4_CKPT="${SEQFT_ROOT}/20260808_130308+cl4_t9_2k_4gpu_bs32_ga2/final_model/pytorch_model.pt"

for required in \
    "${ANALYSIS_SCRIPT}" \
    "${BASE_CKPT}" \
    "${CL1_CKPT}" \
    "${CL2_CKPT}" \
    "${CL3_CKPT}" \
    "${CL4_CKPT}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Missing required file:"
        echo "        ${required}"
        exit 1
    fi
done

ANALYSIS_GPU="${ANALYSIS_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${ANALYSIS_GPU}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ROOT}/results/analysis/lawam_cl/libero_goal/seqft_transition_update_rank/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/analysis.log"

cat > "${OUTPUT_DIR}/lora_sr_reference.csv" <<'CSV'
task,task_id,flow_lora_r8_sr
T6,6,0.70
T7,7,0.98
T8,8,0.86
T9,9,0.34
CSV

cat > "${OUTPUT_DIR}/PROTOCOL.txt" <<EOF
Goal SeqFT transition-wise Flow update localization and intrinsic-rank analysis.

IMPORTANT:
  This is an exploratory structural diagnostic using the original sequential
  SeqFT checkpoints. It is NOT identical to the deleted Frozen-Upstream
  independent Full-Flow oracle checkpoints.

We analyze the Flow update introduced during each SeqFT task transition:

  T6: Base -> CL1
      ${BASE_CKPT}
      ${CL1_CKPT}

  T7: CL1 -> CL2
      ${CL1_CKPT}
      ${CL2_CKPT}

  T8: CL2 -> CL3
      ${CL2_CKPT}
      ${CL3_CKPT}

  T9: CL3 -> CL4
      ${CL3_CKPT}
      ${CL4_CKPT}

Why adjacent transitions:
  SeqFT is sequentially inherited. Comparing Base directly with CL2/CL3/CL4
  would mix updates from multiple tasks. Adjacent-stage differences better
  approximate the Flow update induced while learning the current task.

Caveat:
  Upstream modules also change in SeqFT, so each Flow update is co-adapted with
  a changing upstream. Therefore these results should be used to diagnose
  action-head update structure (target location / intrinsic rank), not as a
  strict causal decomposition of the Frozen-Upstream LoRA performance.

Current Flow-LoRA v1:
  rank = 8
  targets = 16 blocks x (Q/K/V/O + FFN-in/out) = 96 matrices

Observed Flow-LoRA r8 single-task SR:
  T6=0.70
  T7=0.98
  T8=0.86
  T9=0.34
EOF

echo "==========================================================" | tee "${LOG_FILE}"
echo " LaWAM Goal SeqFT Transition Update + Rank Analysis" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"
echo "Analysis GPU : ${ANALYSIS_GPU}" | tee -a "${LOG_FILE}"
echo "Output       : ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "SVD ranks    : 4 8 16 32 64" | tee -a "${LOG_FILE}"
echo "==========================================================" | tee -a "${LOG_FILE}"

run_transition() {
    local label="$1"
    local task_id="$2"
    local before_ckpt="$3"
    local after_ckpt="$4"

    local out="${OUTPUT_DIR}/${label}"
    mkdir -p "${out}"

    echo | tee -a "${LOG_FILE}"
    echo "----------------------------------------------------------" | tee -a "${LOG_FILE}"
    echo " ${label}/T${task_id}: transition analysis" | tee -a "${LOG_FILE}"
    echo " BEFORE: ${before_ckpt}" | tee -a "${LOG_FILE}"
    echo " AFTER : ${after_ckpt}" | tee -a "${LOG_FILE}"
    echo "----------------------------------------------------------" | tee -a "${LOG_FILE}"

    python "${ANALYSIS_SCRIPT}" \
        --base "${before_ckpt}" \
        --task "${label}" "${task_id}" "${after_ckpt}" \
        --ranks 4 8 16 32 64 \
        --device cuda:0 \
        --output-dir "${out}" \
        2>&1 | tee -a "${LOG_FILE}"
}

run_transition "T6" 6 "${BASE_CKPT}" "${CL1_CKPT}"
run_transition "T7" 7 "${CL1_CKPT}" "${CL2_CKPT}"
run_transition "T8" 8 "${CL2_CKPT}" "${CL3_CKPT}"
run_transition "T9" 9 "${CL3_CKPT}" "${CL4_CKPT}"

# Merge the four transition-level outputs into convenient combined CSV files.
python - "${OUTPUT_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
labels = ["T6", "T7", "T8", "T9"]

files = [
    "task_overview.csv",
    "component_update_summary.csv",
    "svd_weighted_summary.csv",
    "svd_per_target_tensor.csv",
    "per_tensor_update.csv",
]

for filename in files:
    rows = []
    fieldnames = None
    for label in labels:
        path = root / label / filename
        if not path.is_file():
            raise RuntimeError(f"Missing expected result: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise RuntimeError(
                    f"CSV schema mismatch for {filename}: "
                    f"{reader.fieldnames} != {fieldnames}"
                )
            rows.extend(reader)

    out = root / filename
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Combined compact summary.
overview_path = root / "task_overview.csv"
with overview_path.open("r", encoding="utf-8", newline="") as f:
    overview = list(csv.DictReader(f))

sr = {"T6": 0.70, "T7": 0.98, "T8": 0.86, "T9": 0.34}

summary_path = root / "SUMMARY_COMBINED.txt"
with summary_path.open("w", encoding="utf-8") as f:
    f.write("Goal SeqFT transition-wise Flow Update / Rank Analysis\n")
    f.write("=" * 76 + "\n\n")
    f.write(
        "NOTE: Adjacent SeqFT transitions are used as an exploratory proxy for\n"
        "task-local Flow updates. Upstream also changes during SeqFT.\n\n"
    )

    for row in overview:
        label = row["task"]
        f.write(f"{label}/T{row['task_id']}  |  LoRA-r8 SR={sr.get(label, float('nan')):.2f}\n")
        f.write(
            "  target coverage of total Flow update : "
            f"{100*float(row['current_lora_target_coverage_of_total_flow_update']):.2f}%\n"
        )
        for r in (4, 8, 16, 32, 64):
            cap = float(row[f"target_weighted_energy_capture_r{r}"])
            rep = float(row[f"total_flow_representable_fraction_r{r}"])
            f.write(
                f"  r={r:<2d}: target capture={100*cap:6.2f}%"
                f" | total representable={100*rep:6.2f}%\n"
            )
        f.write("\n")

print(f"[OK] Combined results written to {root}")
print(f"[OK] Main summary: {summary_path}")
PY

echo
echo "=========================================================="
echo " SeqFT transition analysis complete"
echo "=========================================================="
echo "Combined summary:"
echo "  ${OUTPUT_DIR}/SUMMARY_COMBINED.txt"
echo "Task overview:"
echo "  ${OUTPUT_DIR}/task_overview.csv"
echo "Component localization:"
echo "  ${OUTPUT_DIR}/component_update_summary.csv"
echo "SVD weighted summary:"
echo "  ${OUTPUT_DIR}/svd_weighted_summary.csv"
echo "Per-target SVD:"
echo "  ${OUTPUT_DIR}/svd_per_target_tensor.csv"
echo "Per-tensor update:"
echo "  ${OUTPUT_DIR}/per_tensor_update.csv"
echo "Protocol:"
echo "  ${OUTPUT_DIR}/PROTOCOL.txt"
echo "=========================================================="