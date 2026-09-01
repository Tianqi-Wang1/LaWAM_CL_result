#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Goal Flow-LoRA follow-up diagnostic
#
# Existing reference:
#   r=8, alpha=8, lr=1e-4
#   T6=0.70, T7=0.98, T8=0.86, T9=0.34
#
# New experiments:
#   A) r=32, alpha=32, lr=1e-4 : T6 and T9
#   B) r=32, alpha=32, lr=3e-4 : T9 only
#
# alpha is increased together with rank so alpha/r remains 1.0, identical to
# the previous r=8, alpha=8 experiment. This isolates rank/capacity first.
# =============================================================================

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

RUNNER="${ROOT}/scripts/run_libero_goal_flow_lora_oracle_v2.sh"
[ -f "${RUNNER}" ] || { echo "[ERROR] Missing ${RUNNER}"; exit 1; }

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_ROOT="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_lora_r32_capacity_lr/${TIMESTAMP}"
mkdir -p "${SUMMARY_ROOT}"
MASTER_LOG="${SUMMARY_ROOT}/ablation.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

common_env=(
    TRAIN_GPUS="${TRAIN_GPUS}"
    POLICY_GPU="${POLICY_GPU}"
    EVAL_GPU="${EVAL_GPU}"
    PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}"
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}"
    NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS}"
    NUM_TRIALS="${NUM_TRIALS}"
    EVAL_WORKERS="${EVAL_WORKERS}"
    SAVE_VIDEOS="${SAVE_VIDEOS}"
    FLOW_LORA_MODE=cl
    LORA_RANK=32
    LORA_ALPHA=32
    LORA_DROPOUT=0.0
    LORA_TARGET_ATTN=true
    LORA_TARGET_FFN=true
)

echo "=========================================================="
echo " Goal Flow-LoRA r32 Capacity/LR Ablation"
echo "=========================================================="
echo "Train GPUs       : ${TRAIN_GPUS}"
echo "Policy/Eval GPU  : ${POLICY_GPU}/${EVAL_GPU}"
echo "Global batch     : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 4))  (assuming 4 TRAIN_GPUS)"
echo "Steps            : ${MAX_TRAIN_STEPS}"
echo "Trials/task      : ${NUM_TRIALS}"
echo "A: r32/a32/lr1e-4 -> T6,T9"
echo "B: r32/a32/lr3e-4 -> T9"
echo "Summary root     : ${SUMMARY_ROOT}"
echo "=========================================================="

# -----------------------------------------------------------------------------
# A. Capacity test: keep alpha/r=1 and original LoRA LR; only raise rank.
# -----------------------------------------------------------------------------
echo
echo "[A] r=32, alpha=32, lr=1e-4, tasks T6/T9"
env \
    "${common_env[@]}" \
    LORA_LR=0.0001 \
    FLOW_LORA_STAGES="1 4" \
    bash "${RUNNER}"

# -----------------------------------------------------------------------------
# B. Optimization sanity: same r32 capacity, higher LoRA LR on hardest task T9.
# -----------------------------------------------------------------------------
echo
echo "[B] r=32, alpha=32, lr=3e-4, task T9"
env \
    "${common_env[@]}" \
    LORA_LR=0.0003 \
    FLOW_LORA_STAGES="4" \
    bash "${RUNNER}"

# -----------------------------------------------------------------------------
# Locate the two produced result CSVs and make one compact comparison table.
# -----------------------------------------------------------------------------
TAG_A="r32_a32_lr0p0001"
TAG_B="r32_a32_lr0p0003"
ROOT_A="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_lora_oracle/${TAG_A}"
ROOT_B="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_lora_oracle/${TAG_B}"

CSV_A=$(find "${ROOT_A}" -type f -name flow_lora_single_task_sr.csv | sort | tail -n 1)
CSV_B=$(find "${ROOT_B}" -type f -name flow_lora_single_task_sr.csv | sort | tail -n 1)

[ -n "${CSV_A}" ] && [ -f "${CSV_A}" ] || { echo "[ERROR] Missing A result CSV"; exit 1; }
[ -n "${CSV_B}" ] && [ -f "${CSV_B}" ] || { echo "[ERROR] Missing B result CSV"; exit 1; }

python - "${CSV_A}" "${CSV_B}" "${SUMMARY_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

a_path = Path(sys.argv[1])
b_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

rows_a = read_rows(a_path)
rows_b = read_rows(b_path)

# Existing r8 measurements are included only as a reference row; they are not
# re-run by this wrapper.
rows = [
    {
        "setting": "r8_a8_lr1e-4_reference",
        "rank": 8,
        "alpha": 8,
        "lr": 1e-4,
        "task_id": 6,
        "success_rate": 0.70,
        "source": "existing formal experiment",
    },
    {
        "setting": "r8_a8_lr1e-4_reference",
        "rank": 8,
        "alpha": 8,
        "lr": 1e-4,
        "task_id": 9,
        "success_rate": 0.34,
        "source": "existing formal experiment",
    },
]

for r in rows_a:
    rows.append(
        {
            "setting": "r32_a32_lr1e-4",
            "rank": 32,
            "alpha": 32,
            "lr": 1e-4,
            "task_id": int(r["task_id"]),
            "success_rate": float(r["success_rate"]),
            "source": str(a_path),
        }
    )

for r in rows_b:
    rows.append(
        {
            "setting": "r32_a32_lr3e-4",
            "rank": 32,
            "alpha": 32,
            "lr": 3e-4,
            "task_id": int(r["task_id"]),
            "success_rate": float(r["success_rate"]),
            "source": str(b_path),
        }
    )

out = out_dir / "comparison.csv"
with out.open("w", encoding="utf-8", newline="") as f:
    fieldnames = ["setting", "rank", "alpha", "lr", "task_id", "success_rate", "source"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("\nCompact comparison")
print("Setting                 | Task | SR")
print("------------------------+------+------")
for r in rows:
    print(f"{r['setting']:<23s} | T{r['task_id']:<3d} | {r['success_rate']:.4f}")
print(f"\n[OK] Saved {out}")
PY

cat > "${SUMMARY_ROOT}/PROTOCOL.txt" <<EOF2
Goal Flow-LoRA capacity / optimization diagnostic.

Reference already available:
  r=8, alpha=8, alpha/r=1, lr=1e-4
  T6=0.70, T9=0.34

Experiment A (capacity):
  r=32, alpha=32, alpha/r=1, lr=1e-4
  T6 and T9 only
  Purpose: change rank/capacity while preserving LoRA scaling and optimization LR.

Experiment B (optimization sanity):
  r=32, alpha=32, alpha/r=1, lr=3e-4
  T9 only
  Purpose: test whether the difficult T9 case benefits from a larger LoRA LR
  after giving the adapter substantially more representational capacity.

All runs:
  initialize independently from the SAME formal Goal Base checkpoint;
  freeze all Base WAM + Base Flow tensors;
  train only the 96 Attention/FFN LoRA A/B pairs;
  use oracle task-ID only for evaluation;
  ${MAX_TRAIN_STEPS} steps, batch/GPU=${PER_DEVICE_BATCH_SIZE}, GA=${GRADIENT_ACCUMULATION_STEPS};
  ${NUM_TRIALS} evaluation rollouts per task.
EOF2

echo
echo "=========================================================="
echo " Ablation complete"
echo "=========================================================="
echo "Comparison: ${SUMMARY_ROOT}/comparison.csv"
echo "Protocol  : ${SUMMARY_ROOT}/PROTOCOL.txt"
echo "Log       : ${MASTER_LOG}"
echo "=========================================================="