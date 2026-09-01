#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

MODE="${MODE:-e1}"
VLM_LORA_RANK="${VLM_LORA_RANK:-32}"
VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
FLOW_LORA_RANK="${FLOW_LORA_RANK:-32}"
FLOW_LORA_ALPHA="${FLOW_LORA_ALPHA:-32}"
CONDITIONING_BOTTLENECK="${CONDITIONING_BOTTLENECK:-128}"
ACTION_LR="${ACTION_LR:-0.0001}"
VLM_LR="${VLM_LR:-0.0001}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DO_EVAL="${DO_EVAL:-true}"

case "${MODE}" in
  e1)
    LABEL="E1 Last8-Dense + VLM-Text-LoRA-r${VLM_LORA_RANK}"
    RUN_TAG="e1_last8_dense_text_lora_r${VLM_LORA_RANK}"
    DENSE_LAYERS="[8,9,10,11,12,13,14,15]"
    TRAIN_E1=true; TRAIN_E3=false; TRAIN_E7=false
    ADAPTER_BOTTLENECK=0
    EXPECTED_PARAMS=135356416
    ;;
  e3)
    LABEL="E3 VLM-Text-LoRA-r${VLM_LORA_RANK} + Action-DiT16-LoRA-r${FLOW_LORA_RANK}"
    RUN_TAG="e3_text_lora_r${VLM_LORA_RANK}_action_lora_r${FLOW_LORA_RANK}"
    DENSE_LAYERS="null"
    TRAIN_E1=false; TRAIN_E3=true; TRAIN_E7=false
    ADAPTER_BOTTLENECK=0
    EXPECTED_PARAMS=29229056
    ;;
  e7)
    LABEL="E7 VLM-Text-LoRA-r${VLM_LORA_RANK} + CLARE-Conditioning-r${CONDITIONING_BOTTLENECK}"
    RUN_TAG="e7_text_lora_r${VLM_LORA_RANK}_conditioning_r${CONDITIONING_BOTTLENECK}"
    DENSE_LAYERS="null"
    TRAIN_E1=false; TRAIN_E3=false; TRAIN_E7=true
    ADAPTER_BOTTLENECK="${CONDITIONING_BOTTLENECK}"
    EXPECTED_PARAMS=26574848
    ;;
  *) echo "[ERROR] MODE must be e1, e3, or e7"; exit 2 ;;
esac

if [ "${VLM_LORA_RANK}" != "32" ] || [ "${VLM_LORA_ALPHA}" != "32" ]; then
  echo "[WARN] v6 verifier's exact parameter budget is defined for VLM r=32/alpha=32."
fi
if [ "${MODE}" = "e3" ] && { [ "${FLOW_LORA_RANK}" != "32" ] || [ "${FLOW_LORA_ALPHA}" != "32" ]; }; then
  echo "[WARN] E3 exact expected budget assumes Flow r=32/alpha=32."
fi
if [ "${MODE}" = "e7" ] && [ "${CONDITIONING_BOTTLENECK}" != "128" ]; then
  echo "[WARN] E7 exact expected budget assumes conditioning bottleneck=128."
fi

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

find_run() { local root="$1" pattern="$2"; find "${root}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1; }
verify_run() {
  local label="$1" run="$2"; [ -n "${run}" ] || { echo "[ERROR] ${label}: run not found"; exit 1; }
  for f in "${run}/config.yaml" "${run}/dataset_statistics.json" "${run}/final_model/pytorch_model.pt"; do
    [ -f "${f}" ] || { echo "[ERROR] ${label}: missing ${f}"; exit 1; }
  done
  echo "[OK] ${label}: ${run}"
}
if [ -z "${BASE_RUN:-}" ]; then BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2'); fi
verify_run "Formal Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

python - <<'PYCODE'
from starVLA.model.framework.latent_world.runtime.freeze_policy import LatentWorldPolicyFreezeConfig
fields=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required={'train_vlm_text_lora_partial_dense','train_vlm_text_lora_conditioning_adapter','train_vlm_flow_lora'}
missing=sorted(required-fields)
if missing: raise RuntimeError(f'Missing E1/E3/E7 v6 fields: {missing}')
print('[OK] E1/E3/E7 v6 support detected.')
PYCODE

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_e1_e3_e7_v6/${RUN_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"; LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_e1_e3_e7_v6/${RUN_TAG}"
MERGE_SCRIPT="${ROOT}/scripts/merge_lora_checkpoint_v6.py"
VERIFY_SCRIPT="${ROOT}/scripts/verify_t9_e1_e3_e7_v6.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t9_${RUN_TAG}_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " ${LABEL}"
echo "=========================================================="
echo "Base          : ${BASE_CKPT}"
echo "Train GPUs    : ${TRAIN_GPUS}"
echo "Global batch  : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Steps/warmup  : ${MAX_TRAIN_STEPS}/${NUM_WARMUP_STEPS}"
echo "Action/VLM LR : ${ACTION_LR}/${VLM_LR}"
echo "VLM LoRA      : text all retained layers, r=${VLM_LORA_RANK}, alpha=${VLM_LORA_ALPHA}"
echo "Flow LoRA     : r=${FLOW_LORA_RANK}, alpha=${FLOW_LORA_ALPHA} (E3 only)"
echo "Dense layers  : ${DENSE_LAYERS} (E1 only)"
echo "Cond adapter  : bottleneck=${ADAPTER_BOTTLENECK}, enc_vlm + all 16 AdaLN (E7 only)"
echo "Expected task params: ${EXPECTED_PARAMS}"
echo "Eval GPUs     : policy=${POLICY_GPU}, libero=${EVAL_GPU}; trials=${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
case "${MODE}" in e1) export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-30231}";; e3) export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-30232}";; e7) export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-30233}";; esac

bash train_lawam.sh \
  "--run_root_dir=${RUN_ROOT}" \
  "--run_id=${RUN_ID}" \
  "--datasets.vla_data.cl_suite=libero_goal" \
  "--datasets.vla_data.cl_task_ids=[9]" \
  "--datasets.vla_data.use_task_filtered_statistics=false" \
  "--trainer.use_pretrained_dataset_statistics=true" \
  "--trainer.pretrained_checkpoint=${BASE_CKPT}" \
  "--trainer.load_pretrained_policy_flow=true" \
  "--trainer.policy_flow_override_checkpoint=null" \
  "--framework.action_model.flow_cfg.residual_expert_num_blocks=0" \
  "--framework.action_model.flow_cfg.residual_expert_layer_indices=null" \
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=${ADAPTER_BOTTLENECK}" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_enc_vlm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_adanorm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_zero_init=true" \
  "--trainer.freeze.freeze_vlm_all=true" \
  "--trainer.freeze.freeze_act_query=true" \
  "--trainer.freeze.freeze_flow_action_query=true" \
  "--trainer.freeze.train_flow_only=false" \
  "--trainer.freeze.train_flow_lora=false" \
  "--trainer.freeze.train_flow_residual_expert=false" \
  "--trainer.freeze.train_flow_partial_dense=${TRAIN_E1}" \
  "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_flow_interface_dense=false" \
  "--trainer.freeze.train_flow_interface_lora=false" \
  "--trainer.freeze.train_vlm_flow_lora=${TRAIN_E3}" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense=${TRAIN_E1}" \
  "--trainer.freeze.train_vlm_text_lora_conditioning_adapter=${TRAIN_E7}" \
  "--trainer.freeze.vlm_lora_target_text=true" \
  "--trainer.freeze.vlm_lora_text_last_n=0" \
  "--trainer.freeze.vlm_lora_target_vision=false" \
  "--trainer.freeze.vlm_lora_target_merger=false" \
  "--trainer.freeze.vlm_lora_rank=${VLM_LORA_RANK}" \
  "--trainer.freeze.vlm_lora_alpha=${VLM_LORA_ALPHA}" \
  "--trainer.freeze.vlm_lora_dropout=0.0" \
  "--trainer.freeze.flow_lora_rank=${FLOW_LORA_RANK}" \
  "--trainer.freeze.flow_lora_alpha=${FLOW_LORA_ALPHA}" \
  "--trainer.freeze.flow_lora_dropout=0.0" \
  "--trainer.freeze.unfreeze_lam_decoder=false" \
  "--trainer.learning_rate.action_model.lr=${ACTION_LR}" \
  "--trainer.learning_rate.vlm.lr=${VLM_LR}" \
  "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" \
  "--datasets.vla_data.num_workers=${NUM_WORKERS}" \
  "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}" \
  "--datasets.vla_data.persistent_workers=true" \
  "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" \
  "--trainer.max_train_steps=${MAX_TRAIN_STEPS}" \
  "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}" \
  "--trainer.logging_frequency=${LOGGING_FREQUENCY}" \
  "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}" \
  "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}" \
  "--trainer.save_interval=${SAVE_INTERVAL}"

unset CUDA_VISIBLE_DEVICES NUM_PROCESSES MAIN_PROCESS_PORT || true
RUN=$(find_run "${RUN_ROOT}" "*+${RUN_ID}")
verify_run "${LABEL}" "${RUN}"
UNMERGED="${RUN}/final_model/pytorch_model.pt"
MERGED="${RUN}/final_model/pytorch_model_merged.pt"

python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PYCODE'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:a=json.load(f)
with open(sys.argv[2],encoding='utf-8') as f:b=json.load(f)
for tag in a:
    for sec in ('action','state'):
        if a[tag][sec] != b[tag][sec]: raise RuntimeError(f'Normalization changed: {tag}/{sec}')
print('[OK] action/state normalization identical to Base.')
PYCODE

python "${MERGE_SCRIPT}" --input "${UNMERGED}" --output "${MERGED}" --alpha "${VLM_LORA_ALPHA}"
python "${VERIFY_SCRIPT}" --base "${BASE_CKPT}" --unmerged "${UNMERGED}" --merged "${MERGED}" --mode "${MODE}" --rank "${VLM_LORA_RANK}"

if [ "${DO_EVAL,,}" = "true" ]; then
  export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
  export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
  export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
  ALIAS="t9_${RUN_TAG}"
  EVAL_MASTER="${OUTPUT_ROOT}/${STAMP}/${ALIAS}"; mkdir -p "${EVAL_MASTER}"
  SUITES="libero_goal" TASK_IDS="9" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
  NUM_WORKERS="${EVAL_WORKERS}" GPU_IDS="${POLICY_GPU}" EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" OUTPUT_ROOT="${EVAL_MASTER}" LIBERO_CKPT_ALIAS="${ALIAS}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${MERGED}"
  EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  [ -n "${EVAL_DIR}" ] || { echo "[ERROR] Evaluation output missing"; exit 1; }
  SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
  python "${SUMMARY_SCRIPT}" --run-dir "${SUITE_DIR}" --task-ids 9 --expected-trials "${NUM_TRIALS}"
  echo "${SUITE_DIR}/per_task_summary.csv" > "${EXPERIMENT_ROOT}/latest_summary_path.txt"
else
  SUITE_DIR="SKIPPED"
fi

echo "=========================================================="
echo " COMPLETE: ${LABEL}"
echo " Run       : ${RUN}"
echo " Merged    : ${MERGED}"
echo " Summary   : ${SUITE_DIR}/per_task_summary.csv"
echo " Master log: ${MASTER_LOG}"
echo "=========================================================="
