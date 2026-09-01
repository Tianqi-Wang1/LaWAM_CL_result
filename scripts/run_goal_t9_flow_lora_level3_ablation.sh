#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Goal T9 Flow-LoRA Level-3 ablation
#
# Reference:
#   Level-2 = Transformer + enc_vlm + output
#   T9 SR = 0.64
#   params = 2,390,016
#
# New:
#   enc  : Level-2 + action encoder
#   dec  : Level-2 + action decoder
#   full : Level-2 + action encoder + action decoder
#
# Use:
#   ABLATIONS="full"
#   ABLATIONS="enc dec full"
# =============================================================================

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

RUNNER="${ROOT}/scripts/run_libero_goal_flow_lora_t9_level3_variant.sh"

[ -f "${RUNNER}" ] || {
    echo "[ERROR] Missing ${RUNNER}"
    exit 1
}

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

ABLATIONS="${ABLATIONS:-enc dec full}"

run_variant() {
    local name="$1"
    local action_enc="$2"
    local action_dec="$3"
    local tag="$4"

    echo
    echo "################################################################################"
    echo "# T9 LEVEL-3: ${name}"
    echo "################################################################################"

    env \
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
        LORA_TARGET_ACTION_ENCODER="${action_enc}" \
        LORA_TARGET_ACTION_DECODER="${action_dec}" \
        TARGET_TAG="${tag}" \
        bash "${RUNNER}"
}

for ab in ${ABLATIONS}; do
    case "${ab}" in
        enc)
            run_variant \
                "Level-2 + action encoder" \
                true \
                false \
                level3_action_encoder
            ;;
        dec)
            run_variant \
                "Level-2 + action decoder" \
                false \
                true \
                level3_action_decoder
            ;;
        full)
            run_variant \
                "Full Level-3" \
                true \
                true \
                level3_full
            ;;
        *)
            echo "[ERROR] Unknown ABLATIONS item: ${ab}"
            echo "        Valid: enc dec full"
            exit 1
            ;;
    esac
done

TS=$(date +"%Y%m%d_%H%M%S")
OUT="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_lora_level3_t9/${TS}"
mkdir -p "${OUT}"
CSV="${OUT}/comparison.csv"

python - "${CSV}" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])

ckroot = Path(
    "/home/jincai_guo/tianqi/CVPR2027/checkpoints/"
    "lawam_cl/libero_goal/flow_lora_level3_t9"
)

variants = [
    (
        "level2_reference",
        None,
        2_390_016,
        0.64,
    ),
    (
        "level2_plus_action_encoder",
        "r8_a8_lr0p0001_level3_action_encoder",
        2_414_848,
        None,
    ),
    (
        "level2_plus_action_decoder",
        "r8_a8_lr0p0001_level3_action_decoder",
        2_414_848,
        None,
    ),
    (
        "full_level3",
        "r8_a8_lr0p0001_level3_full",
        2_439_680,
        None,
    ),
]

rows = []

for setting, tag, params, reference_sr in variants:
    if reference_sr is not None:
        rows.append(
            {
                "setting": setting,
                "task_id": 9,
                "lora_params": params,
                "success_rate": reference_sr,
                "result_file": "existing Level-2 reference",
            }
        )
        continue

    ptr = ckroot / tag / "LATEST_RESULT.txt"

    if not ptr.is_file():
        continue

    result = Path(
        ptr.read_text(encoding="utf-8").strip()
    )

    if not result.is_file():
        raise RuntimeError(
            f"Stale result pointer: {result}"
        )

    with result.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        data = list(csv.DictReader(f))

    matches = [
        row
        for row in data
        if str(row.get("task_id", "")).strip() == "9"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one T9 row in {result}, "
            f"got {len(matches)}"
        )

    rows.append(
        {
            "setting": setting,
            "task_id": 9,
            "lora_params": params,
            "success_rate": float(
                matches[0]["success_rate"]
            ),
            "result_file": str(result),
        }
    )

with out.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "setting",
            "task_id",
            "lora_params",
            "success_rate",
            "result_file",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("T9 Flow-LoRA Level-3 comparison")
print(
    f"{'Setting':34s} | {'Params':>10s} | {'SR':>6s}"
)
print("-" * 58)

for row in rows:
    print(
        f"{row['setting']:34s} | "
        f"{int(row['lora_params']):10,d} | "
        f"{float(row['success_rate']):6.4f}"
    )

print(f"\n[OK] Saved: {out}")
PY

echo
echo "=========================================================="
echo " Goal T9 Flow-LoRA Level-3 ablation complete"
echo "=========================================================="
echo "Comparison:"
echo "  ${CSV}"
echo "=========================================================="
