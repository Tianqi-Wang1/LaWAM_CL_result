#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

MODE="${MODE:-b1}"
ACTION_LR="${ACTION_LR:-0.0001}"
VLM_LR="${VLM_LR:-0.0001}"
CONDITIONING_BOTTLENECK="${CONDITIONING_BOTTLENECK:-128}"
NONLINEAR_BOTTLENECK="${NONLINEAR_BOTTLENECK:-128}"

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

DENSE_LAYERS="null"
NONLINEAR_LAYERS="null"
TRAIN_PD=false
TRAIN_PD_COND_NL=false
TRAIN_TEXT_PD_COND_NL=false
TRAIN_COND_NL=false
HAS_LORA=false
VLM_LORA_RANK=32
VLM_LORA_ALPHA=32

case "${MODE}" in
  b1)
    LABEL="B1 Last4-Dense + Conditioning-r128 + DiT-Nonlinear[0-11]-r128 (Action-only)"
    RUN_TAG="b1_last4_dense_conditioning_r128_nonlinear_0_11_r128"
    DENSE_LAYERS="[12,13,14,15]"
    NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"
    TRAIN_PD=true; TRAIN_PD_COND_NL=true
    EXPECTED_PARAMS=70660096
    PORT_DEFAULT=30431
    ;;
  b2)
    LABEL="B2 Last4-Dense + Conditioning-r128 + DiT-Nonlinear[0-11]-r128 + VLM-Text-LoRA-r32"
    RUN_TAG="b2_last4_dense_conditioning_r128_nonlinear_0_11_r128_text_lora_r32"
    DENSE_LAYERS="[12,13,14,15]"
    NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"
    TRAIN_PD=true; TRAIN_TEXT_PD_COND_NL=true; HAS_LORA=true
    EXPECTED_PARAMS=90583040
    PORT_DEFAULT=30432
    ;;
  b3)
    LABEL="B3 Last8-Dense + Conditioning-r128 + DiT-Nonlinear[0-7]-r128 (Action-only)"
    RUN_TAG="b3_last8_dense_conditioning_r128_nonlinear_0_7_r128"
    DENSE_LAYERS="[8,9,10,11,12,13,14,15]"
    NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7]"
    TRAIN_PD=true; TRAIN_PD_COND_NL=true
    EXPECTED_PARAMS=126279680
    PORT_DEFAULT=30433
    ;;
  b4)
    LABEL="B4 Conditioning-r128 + DiT-Nonlinear[0-15]-r128 (Action-only, no Dense DiT)"
    RUN_TAG="b4_conditioning_r128_nonlinear_0_15_r128"
    NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"
    TRAIN_COND_NL=true
    EXPECTED_PARAMS=15040512
    PORT_DEFAULT=30434
    ;;
  *) echo "[ERROR] MODE must be b1, b2, b3, or b4"; exit 2 ;;
esac

if [ "${CONDITIONING_BOTTLENECK}" != "128" ] || [ "${NONLINEAR_BOTTLENECK}" != "128" ]; then
  echo "[ERROR] v8 exact budgets/verifier currently require CONDITIONING_BOTTLENECK=128 and NONLINEAR_BOTTLENECK=128"
  exit 2
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
required={
    'train_flow_partial_dense_conditioning_nonlinear_adapter',
    'train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter',
    'train_flow_conditioning_nonlinear_adapter',
}
missing=sorted(required-fields)
if missing: raise RuntimeError(f'Missing nonlinear-action v8 fields: {missing}')
print('[OK] Nonlinear-action v8 support detected.')
PYCODE

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_nonlinear_action_adapters_v8/${RUN_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"; LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_nonlinear_action_adapters_v8/${RUN_TAG}"
MERGE_SCRIPT="${ROOT}/scripts/merge_lora_checkpoint_v6.py"
VERIFY_SCRIPT="${ROOT}/scripts/verify_t9_nonlinear_v8.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t9_${RUN_TAG}_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " ${LABEL}"
echo "=========================================================="
echo "Base                    : ${BASE_CKPT}"
echo "Train GPUs              : ${TRAIN_GPUS}"
echo "Global batch            : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Steps/warmup            : ${MAX_TRAIN_STEPS}/${NUM_WARMUP_STEPS}"
echo "Action/VLM LR           : ${ACTION_LR}/${VLM_LR}"
echo "Dense layers            : ${DENSE_LAYERS}"
echo "Conditioning adapter    : bottleneck=${CONDITIONING_BOTTLENECK}, enc_vlm + all16 AdaLN"
echo "DiT nonlinear layers    : ${NONLINEAR_LAYERS}"
echo "DiT nonlinear adapter   : bottleneck=${NONLINEAR_BOTTLENECK}, attention + FFN"
echo "VLM Text LoRA           : enabled=${HAS_LORA}, r=${VLM_LORA_RANK}, alpha=${VLM_LORA_ALPHA}"
echo "Expected task params    : ${EXPECTED_PARAMS}"
echo "Eval GPUs               : policy=${POLICY_GPU}, libero=${EVAL_GPU}; trials=${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${PORT_DEFAULT}}"

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
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=${CONDITIONING_BOTTLENECK}" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_enc_vlm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_adanorm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_zero_init=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_bottleneck=${NONLINEAR_BOTTLENECK}" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_layer_indices=${NONLINEAR_LAYERS}" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_target_attention=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_target_ffn=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_zero_init=true" \
  "--trainer.freeze.freeze_vlm_all=true" \
  "--trainer.freeze.freeze_act_query=true" \
  "--trainer.freeze.freeze_flow_action_query=true" \
  "--trainer.freeze.train_flow_only=false" \
  "--trainer.freeze.train_flow_lora=false" \
  "--trainer.freeze.train_flow_residual_expert=false" \
  "--trainer.freeze.train_flow_partial_dense=${TRAIN_PD}" \
  "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_flow_interface_dense=false" \
  "--trainer.freeze.train_flow_interface_lora=false" \
  "--trainer.freeze.train_vlm_flow_lora=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense=false" \
  "--trainer.freeze.train_vlm_text_lora_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_adapter=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_nonlinear_adapter=${TRAIN_PD_COND_NL}" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter=${TRAIN_TEXT_PD_COND_NL}" \
  "--trainer.freeze.train_flow_conditioning_nonlinear_adapter=${TRAIN_COND_NL}" \
  "--trainer.freeze.vlm_lora_target_text=true" \
  "--trainer.freeze.vlm_lora_text_last_n=0" \
  "--trainer.freeze.vlm_lora_target_vision=false" \
  "--trainer.freeze.vlm_lora_target_merger=false" \
  "--trainer.freeze.vlm_lora_rank=${VLM_LORA_RANK}" \
  "--trainer.freeze.vlm_lora_alpha=${VLM_LORA_ALPHA}" \
  "--trainer.freeze.vlm_lora_dropout=0.0" \
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

if [ "${HAS_LORA}" = true ]; then
  python "${MERGE_SCRIPT}" --input "${UNMERGED}" --output "${MERGED}" --alpha "${VLM_LORA_ALPHA}"
  EVAL_CKPT="${MERGED}"
else
  MERGED="${UNMERGED}"
  EVAL_CKPT="${UNMERGED}"
fi
python "${VERIFY_SCRIPT}" --base "${BASE_CKPT}" --unmerged "${UNMERGED}" --merged "${MERGED}" --mode "${MODE}"

if [ "${DO_EVAL,,}" = "true" ]; then
  export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
  export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
  export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
  ALIAS="t9_${RUN_TAG}"
  EVAL_MASTER="${OUTPUT_ROOT}/${STAMP}/${ALIAS}"; mkdir -p "${EVAL_MASTER}"
  SUITES="libero_goal" TASK_IDS="9" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
  NUM_WORKERS="${EVAL_WORKERS}" GPU_IDS="${POLICY_GPU}" EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" OUTPUT_ROOT="${EVAL_MASTER}" LIBERO_CKPT_ALIAS="${ALIAS}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${EVAL_CKPT}"
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
echo " Eval ckpt : ${EVAL_CKPT}"
echo " Summary   : ${SUITE_DIR}/per_task_summary.csv"
echo " Master log: ${MASTER_LOG}"
echo "=========================================================="
