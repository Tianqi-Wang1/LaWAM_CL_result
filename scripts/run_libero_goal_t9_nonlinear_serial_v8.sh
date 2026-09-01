#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

export TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
export POLICY_GPU="${POLICY_GPU:-4}"
export EVAL_GPU="${EVAL_GPU:-5}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_TRIALS="${NUM_TRIALS:-50}"
export CONDITIONING_BOTTLENECK="${CONDITIONING_BOTTLENECK:-128}"
export NONLINEAR_BOTTLENECK="${NONLINEAR_BOTTLENECK:-128}"
export DO_EVAL="${DO_EVAL:-true}"

# Override, e.g. VARIANTS="b1 b3 b4".
VARIANTS="${VARIANTS:-b1 b2 b3 b4}"

ROOT_EXP="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_nonlinear_action_adapters_v8"
mkdir -p "${ROOT_EXP}"
RESULTS_CSV="${ROOT_EXP}/results.csv"
echo "variant,successes,trials,success_rate,task_params" > "${RESULTS_CSV}"

params_for() {
  case "$1" in
    b1) echo 70660096;;
    b2) echo 90583040;;
    b3) echo 126279680;;
    b4) echo 15040512;;
    *) echo 0;;
  esac
}
tag_for() {
  case "$1" in
    b1) echo b1_last4_dense_conditioning_r128_nonlinear_0_11_r128;;
    b2) echo b2_last4_dense_conditioning_r128_nonlinear_0_11_r128_text_lora_r32;;
    b3) echo b3_last8_dense_conditioning_r128_nonlinear_0_7_r128;;
    b4) echo b4_conditioning_r128_nonlinear_0_15_r128;;
  esac
}

idx=0
total=$(wc -w <<< "${VARIANTS}" | tr -d ' ')
for mode in ${VARIANTS}; do
  idx=$((idx+1))
  echo "================ SERIAL ${idx}/${total}: ${mode^^} ================"
  MODE="${mode}" bash scripts/run_libero_goal_t9_nonlinear_variant_v8.sh

  if [ "${DO_EVAL,,}" = "true" ]; then
    tag=$(tag_for "${mode}")
    pointer="${ROOT_EXP}/${tag}/latest_summary_path.txt"
    [ -f "${pointer}" ] || { echo "[ERROR] Missing summary pointer: ${pointer}"; exit 1; }
    summary=$(cat "${pointer}")
    [ -f "${summary}" ] || { echo "[ERROR] Missing summary CSV: ${summary}"; exit 1; }
    python - "${mode}" "${summary}" "$(params_for "${mode}")" "${RESULTS_CSV}" <<'PY'
import csv,sys
mode,path,params,out=sys.argv[1:]
with open(path,newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
if not rows:
    raise RuntimeError(f'No rows in {path}')
r=rows[0]
def pick(*keys):
    for k in keys:
        if k in r and r[k] not in ('',None): return r[k]
    raise KeyError(keys)
succ=pick('successes','success','Success')
trials=pick('trials','Trials')
sr=pick('success_rate','sr','SR')
with open(out,'a',newline='',encoding='utf-8') as f:
    csv.writer(f).writerow([mode,succ,trials,sr,params])
print(f'[OK] appended {mode}: SR={sr}')
PY
  fi
done

echo "=========================================================="
echo " Nonlinear action-adapter v8 serial protocol complete."
echo " Results: ${RESULTS_CSV}"
echo "=========================================================="
cat "${RESULTS_CSV}"
