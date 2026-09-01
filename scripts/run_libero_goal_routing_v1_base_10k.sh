#!/usr/bin/env bash
set -euo pipefail

# Routing-V1 formal new Base: T0-T5, 10K optimizer steps, starting from the released LaWAM pretrain.
# VLM (the retained first-16-layer backbone), LA queries/QFormer, LaWM decoder,
# full Action Expert, and the new z* head are trainable. Stage-1 IDM teacher and
# DINO feature extractor remain frozen by the existing LaWAM implementation.

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${ROOT}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt}"
ROUTING_ROOT="${ROUTING_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_base10k_expert2k}"
RUN_ROOT="${ROUTING_ROOT}/base_runs"
LOG_ROOT="${ROUTING_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_base10k_expert2k/base"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-600}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
# Disable periodic checkpoints by default; trainer still writes final_model at the end.
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
DO_EVAL="${DO_EVAL:-false}"
ACTION_LR="${ACTION_LR:-0.0001}"
VLM_LR="${VLM_LR:-0.0001}"
WORLD_LR="${WORLD_LR:-0.0001}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
WORLD_WEIGHT="${WORLD_WEIGHT:-0.1}"
LATENT_HEAD_HIDDEN="${LATENT_HEAD_HIDDEN:-1024}"

[ -f "${PRETRAIN_CKPT}" ] || { echo "[ERROR] Missing pretrain: ${PRETRAIN_CKPT}"; exit 1; }

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="routing_v1_base_t0_5_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " Routing-V1 FORMAL NEW BASE / LIBERO-Goal T0-T5 / 10K"
echo " Init             : ${PRETRAIN_CKPT}"
echo " Steps            : ${MAX_TRAIN_STEPS}"
echo " Global batch     : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo " VLM/World/Action : trainable (${VLM_LR}/${WORLD_LR}/${ACTION_LR})"
echo " Query/QFormer    : trainable"
echo " z* loss          : MSE(z*, z_GT), weight=${LATENT_WEIGHT}"
echo " h* loss          : MSE(LaWM(h_t,z*), h_GT), weight=${WORLD_WEIGHT}"
echo " LaWM             : trainable in Base"
echo " IDM/DINO         : frozen teacher/features"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-31860}"

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
  "--framework.action_model.enable_expert_latent_aux=true" \
  "--framework.action_model.expert_latent_distill_weight=${LATENT_WEIGHT}" \
  "--framework.action_model.expert_world_loss_weight=${WORLD_WEIGHT}" \
  "--framework.action_model.expert_latent_loss_type=mse" \
  "--framework.action_model.flow_cfg.enable_expert_latent_head=true" \
  "--framework.action_model.flow_cfg.expert_latent_dim=-1" \
  "--framework.action_model.flow_cfg.expert_latent_head_hidden_dim=${LATENT_HEAD_HIDDEN}" \
  "--framework.action_model.flow_cfg.expert_latent_head_dropout=0.0" \
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
  "--trainer.freeze.train_flow_only=false" \
  "--trainer.freeze.train_flow_lora=false" \
  "--trainer.freeze.train_flow_residual_expert=false" \
  "--trainer.freeze.train_flow_partial_dense=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_expert_latent_head=false" \
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
for f in "${RUN}/config.yaml" "${RUN}/dataset_statistics.json" "${RUN}/final_model/pytorch_model.pt"; do
  [ -f "${f}" ] || { echo "[ERROR] Missing ${f}"; exit 1; }
done
BASE_CKPT="${RUN}/final_model/pytorch_model.pt"
python scripts/verify_routing_v1_checkpoint.py \
  --checkpoint "${BASE_CKPT}" --config "${RUN}/config.yaml" --mode base

echo "${RUN}" > "${ROUTING_ROOT}/latest_base_run.txt"

if [ "${DO_EVAL,,}" = "true" ]; then
  export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
  export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
  export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
  ALIAS="routing_v1_base_t0_5"
  EVAL_MASTER="${OUTPUT_ROOT}/${STAMP}/${ALIAS}"; mkdir -p "${EVAL_MASTER}"
  SUITES="libero_goal" TASK_IDS="0 1 2 3 4 5" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
  NUM_WORKERS="${EVAL_WORKERS}" GPU_IDS="${POLICY_GPU}" EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" OUTPUT_ROOT="${EVAL_MASTER}" LIBERO_CKPT_ALIAS="${ALIAS}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${BASE_CKPT}"
  EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
  python scripts/summarize_libero_cl_eval.py --run-dir "${SUITE_DIR}" --task-ids 0 1 2 3 4 5 --expected-trials "${NUM_TRIALS}"
fi

echo "=========================================================="
echo " COMPLETE Routing-V1 Base"
echo " Run       : ${RUN}"
echo " Checkpoint: ${BASE_CKPT}"
echo " Log       : ${MASTER_LOG}"
echo "=========================================================="
