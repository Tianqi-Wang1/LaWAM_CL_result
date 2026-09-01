#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Goal T9 Flow-LoRA Level-2 ablation
#
# Existing reference:
#   Transformer Q/K/V/O + FFN, r8/a8/lr1e-4 -> T9 SR = 0.34
#
# New experiments:
#   encvlm : Transformer + enc_vlm
#   output : Transformer + proj_out_1 / proj_out_2
#   both   : Transformer + enc_vlm + proj_out_1 / proj_out_2
#
# Select subset with:
#   ABLATIONS="both"
#   ABLATIONS="encvlm output"
# =============================================================================

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

RUNNER="${ROOT}/scripts/run_libero_goal_flow_lora_t9_level2_variant.sh"

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

ABLATIONS="${ABLATIONS:-encvlm output both}"

run_variant() {
    local name="$1"
    local enc_vlm="$2"
    local output="$3"
    local tag="$4"

    echo
    echo "################################################################################"
    echo "# T9 LEVEL-2 ABLATION: ${name}"
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
        LORA_TARGET_ENC_VLM="${enc_vlm}" \
        LORA_TARGET_OUTPUT="${output}" \
        TARGET_TAG="${tag}" \
        bash "${RUNNER}"
}

for ab in ${ABLATIONS}; do
    case "${ab}" in
        encvlm)
            run_variant \
                "Transformer + enc_vlm" \
                true \
                false \
                level2_encvlm
            ;;
        output)
            run_variant \
                "Transformer + output" \
                false \
                true \
                level2_output
            ;;
        both)
            run_variant \
                "Transformer + enc_vlm + output" \
                true \
                true \
                level2_encvlm_output
            ;;
        *)
            echo "[ERROR] Unknown ABLATIONS item: ${ab}"
            echo "        Valid: encvlm output both"
            exit 1
            ;;
    esac
done

# -------------------------------------------------------------------------
# Aggregate completed variants + existing baseline into one compact table.
# -------------------------------------------------------------------------
TS=$(date +"%Y%m%d_%H%M%S")
OUT="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_lora_level2_t9/${TS}"
mkdir -p "${OUT}"
CSV="${OUT}/comparison.csv"

python - "${CSV}" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])

ckroot = Path(
    "/home/jincai_guo/tianqi/CVPR2027/checkpoints/"
    "lawam_cl/libero_goal/flow_lora_level2_t9"
)

variants = [
    (
        "transformer_only_reference",
        None,
        0.34,
        2326528,
    ),
    (
        "transformer_plus_enc_vlm",
        "r8_a8_lr0p0001_level2_encvlm",
        None,
        2349056,
    ),
    (
        "transformer_plus_output",
        "r8_a8_lr0p0001_level2_output",
        None,
        2367488,
    ),
    (
        "transformer_plus_enc_vlm_output",
        "r8_a8_lr0p0001_level2_encvlm_output",
        None,
        2390016,
    ),
]

rows = []

for setting, tag, reference_sr, params in variants:
    if reference_sr is not None:
        rows.append(
            {
                "setting": setting,
                "task_id": 9,
                "lora_params": params,
                "success_rate": reference_sr,
                "result_file": "existing r8 Transformer-only reference",
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
        row for row in data
        if str(row.get("task_id", "")).strip() == "9"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one T9 row in {result}, "
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
print("T9 Flow-LoRA Level-2 comparison")
print(
    f"{'Setting':38s} | {'Params':>10s} | {'SR':>6s}"
)
print("-" * 62)

for row in rows:
    print(
        f"{row['setting']:38s} | "
        f"{int(row['lora_params']):10,d} | "
        f"{float(row['success_rate']):6.4f}"
    )

print(f"\n[OK] Saved: {out}")
PY

echo
echo "=========================================================="
echo " Goal T9 Flow-LoRA Level-2 ablation complete"
echo "=========================================================="
echo "Comparison:"
echo "  ${CSV}"
echo "=========================================================="
