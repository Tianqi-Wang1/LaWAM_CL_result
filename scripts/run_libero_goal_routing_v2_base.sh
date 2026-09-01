#!/usr/bin/env bash
set -euo pipefail

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${ROOT}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
RUN_ROOT="${V2_ROOT}/base_runs"; LOG_ROOT="${V2_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"; IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"; NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-600}"
NUM_WORKERS="${NUM_WORKERS:-4}"; VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"; TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
ACTION_LR="${ACTION_LR:-0.0001}"; VLM_LR="${VLM_LR:-0.0001}"; WORLD_LR="${WORLD_LR:-0.0001}"

[ -f "${PRETRAIN_CKPT}" ] || { echo "[ERROR] Missing pretrain: ${PRETRAIN_CKPT}"; exit 1; }
export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 STARVLA_WORKER_OMP_THREADS=1 OMP_NUM_THREADS=1
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="routing_v2_base_t0_5_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"; exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo " Routing-V2 Base T0-T5: ORIGINAL LaWAM objective, NO z*, NO V2 adapters"
echo " Init: ${PRETRAIN_CKPT}"
echo " Steps: ${MAX_TRAIN_STEPS}; global batch=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo " VLM + Base queries + QFormer + full LaWM + full Action Expert trainable"
echo " Evaluation intentionally deferred until ALL T6-T9 skills finish."
echo "======================================================================"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" NUM_PROCESSES="${NUM_GPUS}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-32160}"
bash train_lawam.sh \
  "--run_root_dir=${RUN_ROOT}" \
  "--run_id=${RUN_ID}" \
  "--datasets.vla_data.cl_suite=libero_goal" \
  "--datasets.vla_data.cl_task_ids=[0,1,2,3,4,5]" \
  "--datasets.vla_data.use_task_filtered_statistics=true" \
  "--trainer.pretrained_checkpoint=${PRETRAIN_CKPT}" \
  "--trainer.load_pretrained_policy_flow=true" \
  "--trainer.policy_flow_override_checkpoint=null" \
  "--trainer.use_pretrained_dataset_statistics=false" \
  "--framework.action_model.enable_expert_latent_aux=false" \
  "--framework.action_model.flow_cfg.enable_expert_latent_head=false" \
  "--framework.action_model.routing_v2_enable_query_delta=false" \
  "--framework.action_model.routing_v2_enable_memory=false" \
  "--framework.action_model.routing_v2_memory_train_mode=false" \
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=0" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_bottleneck=0" \
  "--trainer.freeze.freeze_vision_backbone=false" \
  "--trainer.freeze.freeze_llm_backbone=false" \
  "--trainer.freeze.freeze_last_llm_layer=false" \
  "--trainer.freeze.freeze_embedding=false" \
  "--trainer.freeze.unfreeze_vision_merger=true" \
  "--trainer.freeze.keep_llm_first_n_layers=16" \
  "--trainer.freeze.unfreeze_llm_last_n_layers=-1" \
  "--trainer.freeze.freeze_vlm_all=false" \
  "--trainer.freeze.freeze_act_query=false" \
  "--trainer.freeze.freeze_flow_action_query=false" \
  "--trainer.freeze.train_routing_v2_skill=false" \
  "--trainer.freeze.train_routing_v2_memory_only=false" \
  "--trainer.freeze.train_expert_latent_head=false" \
  "--trainer.freeze.train_flow_only=false" \
  "--trainer.freeze.train_flow_lora=false" \
  "--trainer.freeze.train_flow_residual_expert=false" \
  "--trainer.freeze.train_flow_partial_dense=false" \
  "--trainer.freeze.unfreeze_lam_decoder=true" \
  "--trainer.learning_rate.action_model.lr=${ACTION_LR}" \
  "--trainer.learning_rate.world_model.lr=${WORLD_LR}" \
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

RUN=$(find "${RUN_ROOT}" -maxdepth 1 -type d -name "*+${RUN_ID}" | sort | tail -n 1)
[ -n "${RUN}" ] || { echo "[ERROR] Base run not found"; exit 1; }
for f in "${RUN}/config.yaml" "${RUN}/dataset_statistics.json" "${RUN}/final_model/pytorch_model.pt"; do [ -f "${f}" ] || { echo "[ERROR] missing ${f}"; exit 1; }; done
echo "${RUN}" > "${V2_ROOT}/latest_base_run.txt"
echo "[OK] Routing-V2 Base: ${RUN}/final_model/pytorch_model.pt"
