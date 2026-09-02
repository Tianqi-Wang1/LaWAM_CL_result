#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"

TASK_ID="${TASK_ID:?Set TASK_ID=6/7/8/9}"
case "${TASK_ID}" in
  6|7|8|9) ;;
  *) echo "[ERROR] TASK_ID must be one of 6,7,8,9; got ${TASK_ID}"; exit 1 ;;
esac

V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2_b2only}"
TASK_ROOT="${V2_ROOT}/task${TASK_ID}"
SKILL_RUN="${SKILL_RUN:-$(cat "${TASK_ROOT}/latest_skill_run.txt")}"
SKILL_CKPT="${SKILL_RUN}/final_model/pytorch_model.pt"
[ -f "${SKILL_CKPT}" ] || { echo "[ERROR] Missing simplified skill: ${SKILL_CKPT}"; exit 1; }

RUN_ROOT="${TASK_ROOT}/memory_runs"
LOG_ROOT="${TASK_ROOT}/logs"
MEMORY_OUT="${TASK_ROOT}/routing_memory"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${MEMORY_OUT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-250}"
MEMORY_SNAPSHOT_STEPS="${MEMORY_SNAPSHOT_STEPS:-1000,2000,5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
MEMORY_LR="${MEMORY_LR:-0.0003}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-250}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
DENSE_LAYERS="[12,13,14,15]"
NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"

IFS=',' read -ra SNAPSHOT_ARRAY <<< "${MEMORY_SNAPSHOT_STEPS}"
if [ "${#SNAPSHOT_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] MEMORY_SNAPSHOT_STEPS is empty"
  exit 1
fi
LATEST_SNAPSHOT=0
for step in "${SNAPSHOT_ARRAY[@]}"; do
  if ! [[ "${step}" =~ ^[0-9]+$ ]] || [ "${step}" -le 0 ] || [ "${step}" -gt "${MAX_TRAIN_STEPS}" ]; then
    echo "[ERROR] Invalid AE snapshot step ${step} for max=${MAX_TRAIN_STEPS}"
    exit 1
  fi
  if [ "${step}" -ne "${MAX_TRAIN_STEPS}" ] && [ $((step % SAVE_INTERVAL)) -ne 0 ]; then
    echo "[ERROR] Intermediate AE step ${step} must be divisible by SAVE_INTERVAL=${SAVE_INTERVAL}"
    exit 1
  fi
  if [ "${step}" -gt "${LATEST_SNAPSHOT}" ]; then
    LATEST_SNAPSHOT="${step}"
  fi
done

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t${TASK_ID}_routing_v2_b2only_memory_${MAX_TRAIN_STEPS}step"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo " Simplified Routing-V2 Memories / T${TASK_ID}: policy fully frozen"
echo " Skill checkpoint : ${SKILL_CKPT}"
echo " Semantic input   : shared Base VLM + Base queries"
echo " Dynamics input   : task Text-LoRA -> shared QFormer -> shared Base-WM"
echo " Dynamics AE      : [h_t, Delta h], no direct z"
echo " Joint AE steps   : ${MAX_TRAIN_STEPS}; snapshots=${MEMORY_SNAPSHOT_STEPS}"
echo " Full checkpoints : deleted after compact AE extraction"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((33600 + TASK_ID))}"

bash train_lawam.sh \
  "--run_root_dir=${RUN_ROOT}" \
  "--run_id=${RUN_ID}" \
  "--datasets.vla_data.cl_suite=libero_goal" \
  "--datasets.vla_data.cl_task_ids=[${TASK_ID}]" \
  "--datasets.vla_data.use_task_filtered_statistics=false" \
  "--trainer.use_pretrained_dataset_statistics=true" \
  "--trainer.pretrained_checkpoint=${SKILL_CKPT}" \
  "--trainer.load_pretrained_policy_flow=true" \
  "--trainer.policy_flow_override_checkpoint=null" \
  "--framework.action_model.enable_expert_latent_aux=false" \
  "--framework.action_model.flow_cfg.enable_expert_latent_head=false" \
  "--framework.action_model.routing_v2_enable_query_delta=false" \
  "--framework.action_model.routing_v2_enable_memory=true" \
  "--framework.action_model.routing_v2_memory_train_mode=true" \
  "--framework.action_model.routing_v2_memory_dynamics_only=false" \
  "--framework.action_model.routing_v2_semantic_bottleneck=128" \
  "--framework.action_model.routing_v2_dynamics_hidden=192" \
  "--framework.action_model.routing_v2_dynamics_layers=2" \
  "--framework.action_model.routing_v2_dynamics_heads=6" \
  "--framework.action_model.routing_v2_dynamics_ffn=768" \
  "--framework.action_model.routing_v2_dynamics_input_mode=hdh" \
  "--framework.action_model.routing_v2_dynamics_wm_source=base" \
  "--framework.action_model.routing_v2_dynamics_z_weight=0.0" \
  "--framework.action_model.routing_v2_dynamics_pred_weight=0.5" \
  "--framework.action_model.routing_v2_semantic_loss_weight=1.0" \
  "--framework.action_model.routing_v2_dynamics_loss_weight=1.0" \
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=128" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_enc_vlm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_adanorm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_zero_init=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_bottleneck=128" \
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
  "--trainer.freeze.train_flow_partial_dense=false" \
  "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_vlm_flow_lora=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense=false" \
  "--trainer.freeze.train_vlm_text_lora_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_adapter=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_flow_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_routing_v2_skill=false" \
  "--trainer.freeze.train_routing_v2_memory_only=true" \
  "--trainer.freeze.routing_v2_b2_only_skill_path=true" \
  "--trainer.freeze.vlm_lora_target_text=true" \
  "--trainer.freeze.vlm_lora_text_last_n=0" \
  "--trainer.freeze.vlm_lora_target_vision=false" \
  "--trainer.freeze.vlm_lora_target_merger=false" \
  "--trainer.freeze.vlm_lora_rank=32" \
  "--trainer.freeze.vlm_lora_alpha=32" \
  "--trainer.freeze.vlm_lora_dropout=0.0" \
  "--trainer.freeze.unfreeze_lam_decoder=false" \
  "--trainer.learning_rate.base=${MEMORY_LR}" \
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

RUN=$(find "${RUN_ROOT}" -maxdepth 1 -type d -name "*+${RUN_ID}" | sort | tail -n 1)
[ -n "${RUN}" ] || { echo "[ERROR] Memory run not found"; exit 1; }
for file in "${RUN}/config.yaml" "${RUN}/final_model/pytorch_model.pt"; do
  [ -f "${file}" ] || { echo "[ERROR] Missing ${file}"; exit 1; }
done

python scripts/extract_routing_v2_memory_snapshots.py \
  --run "${RUN}" \
  --output-root "${MEMORY_OUT}" \
  --steps "${SNAPSHOT_ARRAY[@]}" \
  --max-step "${MAX_TRAIN_STEPS}" \
  --latest-step "${LATEST_SNAPSHOT}" \
  --config "${RUN}/config.yaml"

echo "${RUN}" > "${TASK_ROOT}/latest_memory_run.txt"

for step in "${SNAPSHOT_ARRAY[@]}"; do
  for file in routing_memory.pt semantic_ae.pt dynamics_ae.pt metadata.json; do
    [ -f "${MEMORY_OUT}/step_${step}/${file}" ] || {
      echo "[ERROR] Missing extracted AE artifact: ${MEMORY_OUT}/step_${step}/${file}"
      exit 1
    }
  done
done
[ -f "${MEMORY_OUT}/manifest.json" ] || { echo "[ERROR] Missing AE manifest"; exit 1; }

if [ "${KEEP_MEMORY_FULL_CHECKPOINTS:-false}" != "true" ]; then
  find "${RUN}/checkpoints" -maxdepth 1 -type f -name '*_pytorch_model.pt' -delete
  rm -f "${RUN}/final_model/pytorch_model.pt"
  echo "[INFO] Removed temporary full policy checkpoints after verified AE extraction."
fi

echo "[OK] T${TASK_ID} AE snapshots saved under ${MEMORY_OUT}"
