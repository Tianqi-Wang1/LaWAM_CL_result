#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Goal T9 Flow-LoRA target ablation
# Existing reference: Transformer Q/K/V/O + FFN, r8/a8/lr1e-4 -> SR 0.34
# New experiments hold rank/alpha/LR/training protocol fixed and add targets.
#
# Optional ABlations selector, e.g.:
#   ABLATIONS="encwm all" bash ...
# Valid names: encvlm encwm output all
# =============================================================================

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
RUNNER="${ROOT}/scripts/run_libero_goal_flow_lora_t9_target_v1.sh"
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
ABLATIONS="${ABLATIONS:-encvlm encwm output all}"

run_variant() {
    local name="$1" enc_vlm="$2" enc_wm="$3" output="$4" tag="$5"
    echo
    echo "################################################################################"
    echo "# T9 TARGET ABLATION: ${name}"
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
        LORA_TARGET_ATTN=true \
        LORA_TARGET_FFN=true \
        LORA_TARGET_ENC_VLM="${enc_vlm}" \
        LORA_TARGET_ENC_WM="${enc_wm}" \
        LORA_TARGET_OUTPUT="${output}" \
        TARGET_TAG="${tag}" \
        bash "${RUNNER}"
}

for ab in ${ABLATIONS}; do
    case "${ab}" in
        encvlm)
            run_variant "Transformer + enc_vlm" true false false tf_encvlm
            ;;
        encwm)
            run_variant "Transformer + enc_wm" false true false tf_encwm
            ;;
        output)
            run_variant "Transformer + output projection" false false true tf_output
            ;;
        all)
            run_variant "Transformer + enc_vlm + enc_wm + output" true true true tf_all_interfaces
            ;;
        *)
            echo "[ERROR] Unknown ablation: ${ab}"
            echo "        Valid: encvlm encwm output all"
            exit 1
            ;;
    esac
done

# Aggregate whatever completed variants are available.
TS=$(date +"%Y%m%d_%H%M%S")
OUT="${ROOT}/results/analysis/lawam_cl/libero_goal/flow_lora_target_ablation/${TS}"
mkdir -p "${OUT}"
CSV="${OUT}/comparison.csv"

python - "${CSV}" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
ckroot = Path("/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_lora_target_ablation")
variants = [
    ("transformer_only_reference", None, 0.34),
    ("transformer_plus_enc_vlm", "r8_a8_lr0p0001_tf_encvlm", None),
    ("transformer_plus_enc_wm", "r8_a8_lr0p0001_tf_encwm", None),
    ("transformer_plus_output", "r8_a8_lr0p0001_tf_output", None),
    ("transformer_plus_all_interfaces", "r8_a8_lr0p0001_tf_all_interfaces", None),
]
rows=[]
for name, tag, reference in variants:
    if reference is not None:
        rows.append({
            "setting": name,
            "task_id": 9,
            "success_rate": reference,
            "result_file": "existing r8 reference",
        })
        continue
    ptr = ckroot / tag / "LATEST_RESULT.txt"
    if not ptr.is_file():
        continue
    result = Path(ptr.read_text(encoding="utf-8").strip())
    if not result.is_file():
        raise RuntimeError(f"Result pointer is stale: {result}")
    with result.open("r", encoding="utf-8", newline="") as f:
        data=list(csv.DictReader(f))
    matches=[r for r in data if str(r.get("task_id","")).strip()=="9"]
    if len(matches)!=1:
        raise RuntimeError(f"Expected exactly one T9 row in {result}, got {len(matches)}")
    r=matches[0]
    rows.append({
        "setting": name,
        "task_id": 9,
        "success_rate": float(r["success_rate"]),
        "result_file": str(result),
    })

with out.open("w", encoding="utf-8", newline="") as f:
    w=csv.DictWriter(f, fieldnames=["setting","task_id","success_rate","result_file"])
    w.writeheader(); w.writerows(rows)

print("T9 Flow-LoRA target ablation")
for r in rows:
    print(f"  {r['setting']:35s}: {float(r['success_rate']):.4f}")
print(f"[OK] Saved: {out}")
PY

echo
echo "=========================================================="
echo " Goal T9 target ablation complete"
echo "=========================================================="
echo "Comparison: ${CSV}"
echo "=========================================================="
