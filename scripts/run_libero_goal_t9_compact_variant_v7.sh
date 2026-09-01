#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

MODE="${MODE:-a1}"
ACTION_LR="${ACTION_LR:-0.0001}"
VLM_LR="${VLM_LR:-0.0001}"
CONDITIONING_BOTTLENECK="${CONDITIONING_BOTTLENECK:-128}"

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

# Default mode flags.
DENSE_LAYERS="null"
ADAPTER_BOTTLENECK=0
TRAIN_PD=false
TRAIN_PD_COND=false
TRAIN_TEXT_PD=false
TRAIN_TEXT_PD_COND=false
TRAIN_JOINT_LORA=false
VLM_LORA_RANK=32
VLM_LORA_ALPHA=32
FLOW_LORA_RANK=32
FLOW_LORA_ALPHA=32
HAS_LORA=false

case "${MODE}" in
  a1)
    LABEL="A1 Last4-Dense + Conditioning-r128 (Action-only)"
    RUN_TAG="a1_last4_dense_conditioning_r128"
    DENSE_LAYERS="[12,13,14,15]"
    TRAIN_PD=true; TRAIN_PD_COND=true
    ADAPTER_BOTTLENECK="${CONDITIONING_BOTTLENECK}"
    EXPECTED_PARAMS=64368640
    PORT_DEFAULT=30331
    ;;
  a2)
    LABEL="A2 Last8-Dense + Conditioning-r128 (Action-only)"
    RUN_TAG="a2_last8_dense_conditioning_r128"
    DENSE_LAYERS="[8,9,10,11,12,13,14,15]"
    TRAIN_PD=true; TRAIN_PD_COND=true
    ADAPTER_BOTTLENECK="${CONDITIONING_BOTTLENECK}"
    EXPECTED_PARAMS=122085376
    PORT_DEFAULT=30332
    ;;
  a3)
    LABEL="A3 Last4-Dense + VLM-Text-LoRA-r32"
    RUN_TAG="a3_last4_dense_text_lora_r32"
    DENSE_LAYERS="[12,13,14,15]"
    TRAIN_PD=true; TRAIN_TEXT_PD=true
    HAS_LORA=true
    EXPECTED_PARAMS=77639680
    PORT_DEFAULT=30333
    ;;
  a4)
    LABEL="A4 Last4-Dense + Conditioning-r128 + VLM-Text-LoRA-r32"
    RUN_TAG="a4_last4_dense_conditioning_r128_text_lora_r32"
    DENSE_LAYERS="[12,13,14,15]"
    TRAIN_PD=true; TRAIN_TEXT_PD_COND=true
    ADAPTER_BOTTLENECK="${CONDITIONING_BOTTLENECK}"
    HAS_LORA=true
    EXPECTED_PARAMS=84291584
    PORT_DEFAULT=30334
    ;;
  a5)
    LABEL="A5 VLM-Text-LoRA-r128 + Action-DiT16-LoRA-r128"
    RUN_TAG="a5_text_lora_r128_action_lora_r128"
    TRAIN_JOINT_LORA=true
    VLM_LORA_RANK=128; VLM_LORA_ALPHA=128
    FLOW_LORA_RANK=128; FLOW_LORA_ALPHA=128
    HAS_LORA=true
    EXPECTED_PARAMS=116916224
    PORT_DEFAULT=30335
    ;;
  *) echo "[ERROR] MODE must be a1, a2, a3, a4, or a5"; exit 2 ;;
esac

if { [ "${MODE}" = "a1" ] || [ "${MODE}" = "a2" ] || [ "${MODE}" = "a4" ]; } && [ "${CONDITIONING_BOTTLENECK}" != "128" ]; then
  echo "[ERROR] v7 exact budgets/verifier currently require CONDITIONING_BOTTLENECK=128"; exit 2
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
    'train_flow_partial_dense_conditioning_adapter',
    'train_vlm_text_lora_partial_dense_conditioning_adapter',
    'train_vlm_text_lora_partial_dense',
    'train_vlm_flow_lora',
}
missing=sorted(required-fields)
if missing: raise RuntimeError(f'Missing compact-expert v7 fields: {missing}')
print('[OK] Compact-expert v7 support detected.')
PYCODE

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_compact_action_experts_v7/${RUN_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"; LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_compact_action_experts_v7/${RUN_TAG}"
MERGE_SCRIPT="${ROOT}/scripts/merge_lora_checkpoint_v6.py"
VERIFY_SCRIPT="${ROOT}/scripts/verify_t9_compact_v7.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t9_${RUN_TAG}_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " ${LABEL}"
echo "=========================================================="
echo "Base             : ${BASE_CKPT}"
echo "Train GPUs       : ${TRAIN_GPUS}"
echo "Global batch     : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Steps/warmup     : ${MAX_TRAIN_STEPS}/${NUM_WARMUP_STEPS}"
echo "Action/VLM LR    : ${ACTION_LR}/${VLM_LR}"
echo "Dense layers     : ${DENSE_LAYERS}"
echo "Conditioning     : bottleneck=${ADAPTER_BOTTLENECK}, enc_vlm + all 16 AdaLN"
echo "VLM Text LoRA    : enabled=$([[ ${TRAIN_TEXT_PD} == true || ${TRAIN_TEXT_PD_COND} == true || ${TRAIN_JOINT_LORA} == true ]] && echo true || echo false), r=${VLM_LORA_RANK}, alpha=${VLM_LORA_ALPHA}"
echo "Action DiT LoRA  : enabled=${TRAIN_JOINT_LORA}, r=${FLOW_LORA_RANK}, alpha=${FLOW_LORA_ALPHA}"
echo "Expected task params: ${EXPECTED_PARAMS}"
echo "Eval GPUs        : policy=${POLICY_GPU}, libero=${EVAL_GPU}; trials=${NUM_TRIALS}"
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
  "--trainer.freeze.train_flow_partial_dense=${TRAIN_PD}" \
  "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_flow_interface_dense=false" \
  "--trainer.freeze.train_flow_interface_lora=false" \
  "--trainer.freeze.train_vlm_flow_lora=${TRAIN_JOINT_LORA}" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense=${TRAIN_TEXT_PD}" \
  "--trainer.freeze.train_vlm_text_lora_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_adapter=${TRAIN_PD_COND}" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_adapter=${TRAIN_TEXT_PD_COND}" \
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

if [ "${HAS_LORA}" = true ]; then
  # All LoRA-bearing v7 modes use one common alpha within the checkpoint:
  # A3/A4 -> 32; A5 -> 128 for both VLM and Action LoRA.
  python "${MERGE_SCRIPT}" --input "${UNMERGED}" --output "${MERGED}" --alpha "${VLM_LORA_ALPHA}"
  EVAL_CKPT="${MERGED}"
else
  # No duplication of the ~2.5B checkpoint for action-only modes.
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
