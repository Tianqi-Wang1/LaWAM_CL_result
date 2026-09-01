#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam
ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"; cd "${ROOT}"
TASK_ID="${TASK_ID:?Set TASK_ID=6/7/8/9}"
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
TASK_ROOT="${V2_ROOT}/task${TASK_ID}"
SKILL_RUN="${SKILL_RUN:-$(cat "${TASK_ROOT}/latest_skill_run.txt")}"; SKILL_CKPT="${SKILL_RUN}/final_model/pytorch_model.pt"
[ -f "${SKILL_CKPT}" ] || { echo "[ERROR] Missing skill checkpoint: ${SKILL_CKPT}"; exit 1; }
RUN_ROOT="${TASK_ROOT}/memory_runs"; LOG_ROOT="${TASK_ROOT}/logs"; MEMORY_OUT="${TASK_ROOT}/routing_memory"; mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${MEMORY_OUT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"; IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"; NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"; GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"; NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-50}"
MEMORY_LR="${MEMORY_LR:-0.0003}"; NUM_WORKERS="${NUM_WORKERS:-4}"; VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"; TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-250}"; TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"; SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
DENSE_LAYERS="[12,13,14,15]"; NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"

export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 STARVLA_WORKER_OMP_THREADS=1 OMP_NUM_THREADS=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t${TASK_ID}_routing_v2_memory_${MAX_TRAIN_STEPS}step"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"; exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo " Routing-V2 Memory / T${TASK_ID}: policy FROZEN, train two AEs only"
echo " Semantic AE input : Base VLM + Base query H_act, token-wise bottleneck=128"
echo " Dynamics AE input : [h_t, Delta h, z], 2-layer spatial Transformer hidden=192"
echo " Dynamics mix      : 0.5 GT transition + 0.5 correct-skill predicted transition"
echo " Skill checkpoint  : ${SKILL_CKPT}"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" NUM_PROCESSES="${NUM_GPUS}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((32600 + TASK_ID))}"
bash train_lawam.sh \
  "--run_root_dir=${RUN_ROOT}" "--run_id=${RUN_ID}" \
  "--datasets.vla_data.cl_suite=libero_goal" "--datasets.vla_data.cl_task_ids=[${TASK_ID}]" \
  "--datasets.vla_data.use_task_filtered_statistics=false" "--trainer.use_pretrained_dataset_statistics=true" \
  "--trainer.pretrained_checkpoint=${SKILL_CKPT}" "--trainer.load_pretrained_policy_flow=true" "--trainer.policy_flow_override_checkpoint=null" \
  "--framework.action_model.enable_expert_latent_aux=false" "--framework.action_model.flow_cfg.enable_expert_latent_head=false" \
  "--framework.action_model.routing_v2_enable_query_delta=true" \
  "--framework.action_model.routing_v2_enable_memory=true" "--framework.action_model.routing_v2_memory_train_mode=true" \
  "--framework.action_model.routing_v2_semantic_bottleneck=128" \
  "--framework.action_model.routing_v2_dynamics_hidden=192" "--framework.action_model.routing_v2_dynamics_layers=2" "--framework.action_model.routing_v2_dynamics_heads=6" "--framework.action_model.routing_v2_dynamics_ffn=768" \
  "--framework.action_model.routing_v2_dynamics_z_weight=0.5" "--framework.action_model.routing_v2_dynamics_pred_weight=0.5" \
  "--framework.action_model.routing_v2_semantic_loss_weight=1.0" "--framework.action_model.routing_v2_dynamics_loss_weight=1.0" \
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=128" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_enc_vlm=true" "--framework.action_model.flow_cfg.conditioning_adapter_target_adanorm=true" "--framework.action_model.flow_cfg.conditioning_adapter_zero_init=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_bottleneck=128" "--framework.action_model.flow_cfg.dit_nonlinear_adapter_layer_indices=${NONLINEAR_LAYERS}" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_target_attention=true" "--framework.action_model.flow_cfg.dit_nonlinear_adapter_target_ffn=true" "--framework.action_model.flow_cfg.dit_nonlinear_adapter_zero_init=true" \
  "--trainer.freeze.freeze_vlm_all=true" "--trainer.freeze.freeze_act_query=true" "--trainer.freeze.freeze_flow_action_query=true" \
  "--trainer.freeze.train_flow_partial_dense=false" "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_routing_v2_skill=false" "--trainer.freeze.train_routing_v2_memory_only=true" \
  "--trainer.freeze.vlm_lora_target_text=true" "--trainer.freeze.vlm_lora_text_last_n=0" "--trainer.freeze.vlm_lora_target_vision=false" "--trainer.freeze.vlm_lora_target_merger=false" \
  "--trainer.freeze.vlm_lora_rank=32" "--trainer.freeze.vlm_lora_alpha=32" "--trainer.freeze.vlm_lora_dropout=0.0" \
  "--trainer.freeze.routing_v2_qformer_lora_rank=32" "--trainer.freeze.routing_v2_qformer_lora_alpha=32" "--trainer.freeze.routing_v2_qformer_lora_dropout=0.0" \
  "--trainer.freeze.routing_v2_lawm_lora_rank=32" "--trainer.freeze.routing_v2_lawm_lora_alpha=32" "--trainer.freeze.routing_v2_lawm_lora_dropout=0.0" \
  "--trainer.freeze.unfreeze_lam_decoder=false" \
  "--trainer.learning_rate.base=${MEMORY_LR}" \
  "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" "--datasets.vla_data.num_workers=${NUM_WORKERS}" "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}" "--datasets.vla_data.persistent_workers=true" \
  "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" "--trainer.max_train_steps=${MAX_TRAIN_STEPS}" "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}" \
  "--trainer.logging_frequency=${LOGGING_FREQUENCY}" "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}" "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}" "--trainer.save_interval=${SAVE_INTERVAL}"
unset CUDA_VISIBLE_DEVICES NUM_PROCESSES MAIN_PROCESS_PORT || true
RUN=$(find "${RUN_ROOT}" -maxdepth 1 -type d -name "*+${RUN_ID}" | sort | tail -n 1); [ -n "${RUN}" ] || { echo "[ERROR] memory run not found"; exit 1; }
FULL="${RUN}/final_model/pytorch_model.pt"; [ -f "${FULL}" ] || { echo "[ERROR] missing ${FULL}"; exit 1; }
MEMORY_FILE="${MEMORY_OUT}/routing_memory.pt"
python scripts/extract_routing_v2_memory.py --checkpoint "${FULL}" --output "${MEMORY_FILE}" --config "${RUN}/config.yaml"
cp "${RUN}/config.yaml" "${MEMORY_OUT}/memory_train_config.yaml"
echo "${RUN}" > "${TASK_ROOT}/latest_memory_run.txt"
# Storage-conscious: the task skill checkpoint is already stored separately; keep only the tiny AE memory.
if [ "${KEEP_MEMORY_FULL_CHECKPOINT:-false}" != "true" ]; then
  rm -f "${FULL}"
  echo "[INFO] Removed temporary full memory-phase checkpoint to save storage: ${FULL}"
fi
echo "[OK] T${TASK_ID} routing memory: ${MEMORY_FILE}"
