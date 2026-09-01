#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# T9 Conditioning-Complete ablation.
#
# Default fresh runs:
#   reference : fresh Level-2 r8 rerun
#   adanorm   : reference + AdaNorm LoRA
#   timestep  : reference + timestep encoder LoRA
#   both      : reference + AdaNorm + timestep
#
# Examples:
#   EXPERIMENTS="reference both"
#   EXPERIMENTS="reference adanorm timestep both"
# =============================================================================

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

RUNNER="${ROOT}/scripts/run_libero_goal_flow_conditioning_t9_variant.sh"

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

EXPERIMENTS="${EXPERIMENTS:-reference adanorm timestep both}"

run_one() {
    local mode="$1"

    echo
    echo "################################################################################"
    echo "# T9 CONDITIONING EXPERIMENT: ${mode}"
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
        LORA_RANK=8 \
        LORA_ALPHA=8 \
        LORA_DROPOUT=0.0 \
        LORA_LR=0.0001 \
        bash "${RUNNER}"
}

for exp in ${EXPERIMENTS}; do
    case "${exp}" in
        reference|adanorm|timestep|both)
            run_one "${exp}"
            ;;
        *)
            echo "[ERROR] Unknown experiment: ${exp}"
            echo "        Valid: reference adanorm timestep both"
            exit 1
            ;;
    esac
done

TS=$(date +"%Y%m%d_%H%M%S")
OUT="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_conditioning_complete_t9/${TS}"
mkdir -p "${OUT}"
CSV="${OUT}/comparison.csv"

python - "${CSV}" <<'PY'
import csv,sys
from pathlib import Path

out=Path(sys.argv[1])

ckroot=Path(
    "/home/jincai_guo/tianqi/CVPR2027/checkpoints/"
    "lawam_cl/libero_goal/flow_conditioning_complete_t9"
)

# Keep the historical row only as context; no checkpoint is required.
rows=[
    {
        "setting":"historical_level2_r8",
        "task_id":9,
        "adapter_params":2_390_016,
        "success_rate":0.64,
        "result_file":"historical 50-trial result; checkpoint not required",
    }
]

variants=[
    (
        "fresh_level2_r8_reference",
        "fresh_level2_r8_reference",
        2_390_016,
    ),
    (
        "level2_plus_adanorm",
        "level2_r8_plus_adanorm",
        2_783_232,
    ),
    (
        "level2_plus_timestep",
        "level2_r8_plus_timestep",
        2_416_640,
    ),
    (
        "conditioning_complete",
        "conditioning_complete_r8",
        2_809_856,
    ),
]

for setting,tag,params in variants:
    ptr=ckroot/tag/"LATEST_RESULT.txt"
    if not ptr.is_file():
        continue

    result=Path(ptr.read_text(encoding="utf-8").strip())
    if not result.is_file():
        raise RuntimeError(f"Stale result pointer: {result}")

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
print("T9 Conditioning-LoRA comparison")
print(f"{'Setting':34s} | {'Params':>10s} | {'SR':>6s}")
print("-"*58)
for r in rows:
    print(
        f"{r['setting']:34s} | "
        f"{int(r['adapter_params']):10,d} | "
        f"{float(r['success_rate']):6.4f}"
    )

print(f"\n[OK] Saved: {out}")
PY

echo
echo "=========================================================="
echo " T9 Conditioning-Complete ablation complete"
echo "=========================================================="
echo "Comparison:"
echo "  ${CSV}"
echo "=========================================================="
