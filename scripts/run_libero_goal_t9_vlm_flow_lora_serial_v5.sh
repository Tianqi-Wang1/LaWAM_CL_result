#!/usr/bin/env bash
set -euo pipefail

# Default coarse VLM-location ablation.
# To run the optional text-depth study as well:
#   VARIANTS="text_last4 text_last8 text_all vision merger full" bash ...
VARIANTS="${VARIANTS:-text_all vision merger full}"
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
FLOW_LORA_LR="${FLOW_LORA_LR:-0.0001}"
VLM_LORA_LR="${VLM_LORA_LR:-0.0001}"

STAMP=$(date +"%Y%m%d_%H%M%S")
SERIAL_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_vlm_flow_lora_v5/serial_${STAMP}"
mkdir -p "${SERIAL_ROOT}"
SERIAL_LOG="${SERIAL_ROOT}/serial.log"
RESULT_CSV="${SERIAL_ROOT}/results.csv"
exec > >(tee -a "${SERIAL_LOG}") 2>&1

echo "variant,successes,trials,success_rate,summary_path" > "${RESULT_CSV}"

echo "=========================================================="
echo " T9 SERIAL VLM-LOCATION + FIXED ACTION-DiT16 LoRA-r${LORA_RANK}"
echo "=========================================================="
echo "Variants       : ${VARIANTS}"
echo "Train GPUs     : ${TRAIN_GPUS}"
echo "Eval GPUs      : policy=${POLICY_GPU}, libero=${EVAL_GPU}"
echo "LoRA r/alpha   : ${LORA_RANK}/${LORA_ALPHA}"
echo "Flow/VLM LR    : ${FLOW_LORA_LR}/${VLM_LORA_LR}"
echo "Result CSV     : ${RESULT_CSV}"
echo "=========================================================="

PORT=30141
for variant in ${VARIANTS}; do
  echo
  echo "################################################################"
  echo "# START VARIANT: ${variant}"
  echo "################################################################"
  VLM_VARIANT="${variant}" \
  TRAIN_GPUS="${TRAIN_GPUS}" POLICY_GPU="${POLICY_GPU}" EVAL_GPU="${EVAL_GPU}" \
  LORA_RANK="${LORA_RANK}" LORA_ALPHA="${LORA_ALPHA}" LORA_DROPOUT="${LORA_DROPOUT}" \
  FLOW_LORA_LR="${FLOW_LORA_LR}" VLM_LORA_LR="${VLM_LORA_LR}" \
  MAIN_PROCESS_PORT="${PORT}" \
  bash scripts/run_libero_goal_t9_vlm_flow_lora_variant_v5.sh

  EXP_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_vlm_flow_lora_v5/${variant}"
  PATH_FILE="${EXP_ROOT}/latest_summary_path.txt"
  [ -f "${PATH_FILE}" ] || { echo "[ERROR] Missing summary pointer: ${PATH_FILE}"; exit 1; }
  SUMMARY=$(cat "${PATH_FILE}")
  [ -f "${SUMMARY}" ] || { echo "[ERROR] Missing summary CSV: ${SUMMARY}"; exit 1; }
  python - "${variant}" "${SUMMARY}" "${RESULT_CSV}" <<'PY'
import csv,sys
variant,path,out=sys.argv[1:4]
with open(path,newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
if len(rows)!=1 or int(rows[0]['task_id'])!=9:
    raise RuntimeError(f'Unexpected T9 summary: {rows}')
r=rows[0]
with open(out,'a',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow([variant,r['successes'],r['trials'],r['success_rate'],path])
print(f"[SERIAL] {variant}: {r['successes']}/{r['trials']} = {float(r['success_rate']):.4f}")
PY
  PORT=$((PORT + 1))
done

echo
echo "=========================================================="
echo " SERIAL VLM-LOCATION ABLATION COMPLETE"
echo "=========================================================="
column -s, -t "${RESULT_CSV}" 2>/dev/null || cat "${RESULT_CSV}"
echo "Serial log : ${SERIAL_LOG}"
echo "Result CSV : ${RESULT_CSV}"
echo "=========================================================="
