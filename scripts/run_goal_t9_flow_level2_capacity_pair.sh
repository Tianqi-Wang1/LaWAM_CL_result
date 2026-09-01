#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Run BOTH T9 experiments sequentially.
#
#   r32   : Broad Level-2 LoRA r32/a32
#   dense : Transformer r8 + full-rank zero-init interface deltas
#
# Usage:
#   EXPERIMENTS="r32 dense" ...
#   EXPERIMENTS="dense" ...
# =============================================================================

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

RUNNER="${ROOT}/scripts/run_libero_goal_flow_level2_capacity_variant.sh"

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

EXPERIMENTS="${EXPERIMENTS:-r32 dense}"

run_one() {
    local mode="$1"

    echo
    echo "################################################################################"
    echo "# T9 CAPACITY EXPERIMENT: ${mode}"
    echo "################################################################################"

    env \
        MODE="${mode}" \
        TRAIN_GPUS="${TRAIN_GPUS}" \
        POLICY_GPU="${POLICY_GPU}" \
        EVAL_GPU="${EVAL_GPU}" \
        PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE}" \
        GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
        MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
        NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS}" \
        NUM_TRIALS="${NUM_TRIALS}" \
        EVAL_WORKERS="${EVAL_WORKERS}" \
        SAVE_VIDEOS="${SAVE_VIDEOS}" \
        bash "${RUNNER}"
}

for exp in ${EXPERIMENTS}; do
    case "${exp}" in
        r32)
            run_one broad_r32
            ;;
        dense)
            run_one dense_interface
            ;;
        *)
            echo "[ERROR] Unknown experiment: ${exp}"
            echo "        Valid: r32 dense"
            exit 1
            ;;
    esac
done

TS=$(date +"%Y%m%d_%H%M%S")
OUT="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_level2_capacity_pair/${TS}"
mkdir -p "${OUT}"
CSV="${OUT}/comparison.csv"

python - "${CSV}" <<'PY'
import csv,sys
from pathlib import Path

out=Path(sys.argv[1])
ckroot=Path(
    "/home/jincai_guo/tianqi/CVPR2027/checkpoints/"
    "lawam_cl/libero_goal/flow_level2_capacity_pair"
)

rows=[
    {
        "setting":"transformer_r8_reference",
        "task_id":9,
        "adapter_params":2_326_528,
        "success_rate":0.34,
        "result_file":"existing reference",
    },
    {
        "setting":"level2_r8_reference",
        "task_id":9,
        "adapter_params":2_390_016,
        "success_rate":0.64,
        "result_file":"existing reference",
    },
]

variants=[
    (
        "broad_level2_r32",
        "level2_broad_r32",
        9_560_064,
    ),
    (
        "transformer_r8_dense_interfaces",
        "transformer_r8_dense_interfaces",
        7_045_120,
    ),
]

for setting,tag,params in variants:
    ptr=ckroot/tag/"LATEST_RESULT.txt"
    if not ptr.is_file():
        continue

    result=Path(ptr.read_text(encoding="utf-8").strip())
    if not result.is_file():
        raise RuntimeError(f"Stale pointer: {result}")

    with result.open("r",encoding="utf-8",newline="") as f:
        data=list(csv.DictReader(f))

    matches=[
        r for r in data
        if str(r.get("task_id","")).strip()=="9"
    ]
    if len(matches)!=1:
        raise RuntimeError(
            f"Expected one T9 row in {result}, got {len(matches)}"
        )

    rows.append(
        {
            "setting":setting,
            "task_id":9,
            "adapter_params":params,
            "success_rate":float(matches[0]["success_rate"]),
            "result_file":str(result),
        }
    )

with out.open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(
        f,
        fieldnames=[
            "setting","task_id","adapter_params",
            "success_rate","result_file"
        ],
    )
    w.writeheader()
    w.writerows(rows)

print()
print("T9 Level-2 capacity comparison")
print(f"{'Setting':38s} | {'Params':>10s} | {'SR':>6s}")
print("-"*62)
for r in rows:
    print(
        f"{r['setting']:38s} | "
        f"{int(r['adapter_params']):10,d} | "
        f"{float(r['success_rate']):6.4f}"
    )
print(f"\n[OK] Saved: {out}")
PY

echo
echo "=========================================================="
echo " T9 Level-2 capacity pair complete"
echo "=========================================================="
echo "Comparison:"
echo "  ${CSV}"
echo "=========================================================="
