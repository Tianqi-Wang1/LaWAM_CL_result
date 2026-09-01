#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam
ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"; cd "${ROOT}"
TASK_ID="${TASK_ID:?Set TASK_ID=6/7/8/9}"
case "${TASK_ID}" in 6|7|8|9) ;; *) echo "[ERROR] TASK_ID=${TASK_ID}"; exit 1;; esac
V2_ROOT="${V2_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v2}"
BASE_RUN="${BASE_RUN:-$(cat "${V2_ROOT}/latest_base_run.txt")}"; BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
[ -f "${BASE_CKPT}" ] || { echo "[ERROR] Missing V2 Base: ${BASE_CKPT}"; exit 1; }
BASE_STATS="${BASE_RUN}/dataset_statistics.json"
TASK_ROOT="${V2_ROOT}/task${TASK_ID}"; RUN_ROOT="${TASK_ROOT}/skill_runs"; LOG_ROOT="${TASK_ROOT}/logs"; mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"; IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"; NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"; GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"; NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"; VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"; TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"; TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"; SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
ACTION_LR="${ACTION_LR:-0.0001}"; ADAPTER_LR="${ADAPTER_LR:-0.0001}"
VLM_LORA_RANK="${VLM_LORA_RANK:-32}"; VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
QFORMER_LORA_RANK="${QFORMER_LORA_RANK:-32}"; QFORMER_LORA_ALPHA="${QFORMER_LORA_ALPHA:-32}"
LAWM_LORA_RANK="${LAWM_LORA_RANK:-32}"; LAWM_LORA_ALPHA="${LAWM_LORA_ALPHA:-32}"
COND_BOTTLENECK="${COND_BOTTLENECK:-128}"; NONLINEAR_BOTTLENECK="${NONLINEAR_BOTTLENECK:-128}"
DENSE_LAYERS="[12,13,14,15]"; NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"

export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 STARVLA_WORKER_OMP_THREADS=1 OMP_NUM_THREADS=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t${TASK_ID}_routing_v2_b2_skill_qwm_lora32_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"; exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo " Routing-V2 Skill / T${TASK_ID} -- FRESH from SAME V2 Base"
echo " Base                : ${BASE_CKPT}"
echo " VLM                 : frozen Base + text LoRA r${VLM_LORA_RANK}"
echo " Query               : frozen Base + zero-init task residual (act + flow)"
echo " QFormer             : frozen Base + explicit-Linear LoRA r${QFORMER_LORA_RANK}"
echo " LaWM                : frozen Base + explicit-Linear LoRA r${LAWM_LORA_RANK}"
echo " Action Expert       : B2 last4 dense + cond128 + nonlinear[0:11]128"
echo " z*                  : REMOVED / disabled"
echo " Original LaWAM loss : Lact + 0.1 Ldistill + 0.1 Lwm"
echo " No routing AE loss enters policy training."
echo "======================================================================"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" NUM_PROCESSES="${NUM_GPUS}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((32400 + TASK_ID))}"
bash train_lawam.sh \
  "--run_root_dir=${RUN_ROOT}" "--run_id=${RUN_ID}" \
  "--datasets.vla_data.cl_suite=libero_goal" "--datasets.vla_data.cl_task_ids=[${TASK_ID}]" \
  "--datasets.vla_data.use_task_filtered_statistics=false" \
  "--trainer.use_pretrained_dataset_statistics=true" \
  "--trainer.pretrained_checkpoint=${BASE_CKPT}" "--trainer.load_pretrained_policy_flow=true" "--trainer.policy_flow_override_checkpoint=null" \
  "--framework.action_model.enable_expert_latent_aux=false" "--framework.action_model.flow_cfg.enable_expert_latent_head=false" \
  "--framework.action_model.routing_v2_enable_query_delta=true" \
  "--framework.action_model.routing_v2_enable_memory=false" "--framework.action_model.routing_v2_memory_train_mode=false" \
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=${COND_BOTTLENECK}" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_enc_vlm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_target_adanorm=true" \
  "--framework.action_model.flow_cfg.conditioning_adapter_zero_init=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_bottleneck=${NONLINEAR_BOTTLENECK}" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_layer_indices=${NONLINEAR_LAYERS}" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_target_attention=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_target_ffn=true" \
  "--framework.action_model.flow_cfg.dit_nonlinear_adapter_zero_init=true" \
  "--trainer.freeze.freeze_vlm_all=true" "--trainer.freeze.freeze_act_query=true" "--trainer.freeze.freeze_flow_action_query=true" \
  "--trainer.freeze.train_flow_partial_dense=true" "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_routing_v2_skill=true" "--trainer.freeze.train_routing_v2_memory_only=false" \
  "--trainer.freeze.train_expert_latent_head=false" \
  "--trainer.freeze.vlm_lora_target_text=true" "--trainer.freeze.vlm_lora_text_last_n=0" \
  "--trainer.freeze.vlm_lora_target_vision=false" "--trainer.freeze.vlm_lora_target_merger=false" \
  "--trainer.freeze.vlm_lora_rank=${VLM_LORA_RANK}" "--trainer.freeze.vlm_lora_alpha=${VLM_LORA_ALPHA}" "--trainer.freeze.vlm_lora_dropout=0.0" \
  "--trainer.freeze.routing_v2_qformer_lora_rank=${QFORMER_LORA_RANK}" "--trainer.freeze.routing_v2_qformer_lora_alpha=${QFORMER_LORA_ALPHA}" "--trainer.freeze.routing_v2_qformer_lora_dropout=0.0" \
  "--trainer.freeze.routing_v2_lawm_lora_rank=${LAWM_LORA_RANK}" "--trainer.freeze.routing_v2_lawm_lora_alpha=${LAWM_LORA_ALPHA}" "--trainer.freeze.routing_v2_lawm_lora_dropout=0.0" \
  "--trainer.freeze.unfreeze_lam_decoder=false" \
  "--trainer.learning_rate.base=${ADAPTER_LR}" "--trainer.learning_rate.action_model.lr=${ACTION_LR}" "--trainer.learning_rate.vlm.lr=${ADAPTER_LR}" \
  "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}" "--datasets.vla_data.num_workers=${NUM_WORKERS}" "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}" "--datasets.vla_data.persistent_workers=true" \
  "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}" "--trainer.max_train_steps=${MAX_TRAIN_STEPS}" "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}" \
  "--trainer.logging_frequency=${LOGGING_FREQUENCY}" "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}" "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}" "--trainer.save_interval=${SAVE_INTERVAL}"
unset CUDA_VISIBLE_DEVICES NUM_PROCESSES MAIN_PROCESS_PORT || true
RUN=$(find "${RUN_ROOT}" -maxdepth 1 -type d -name "*+${RUN_ID}" | sort | tail -n 1); [ -n "${RUN}" ] || { echo "[ERROR] skill run not found"; exit 1; }
for f in "${RUN}/config.yaml" "${RUN}/dataset_statistics.json" "${RUN}/final_model/pytorch_model.pt"; do [ -f "${f}" ] || { echo "[ERROR] missing ${f}"; exit 1; }; done
python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:a=json.load(f)
with open(sys.argv[2],encoding='utf-8') as f:b=json.load(f)
for tag in a:
  for sec in ('action','state'):
    if a[tag][sec] != b[tag][sec]: raise RuntimeError(f'Normalization changed: {tag}/{sec}')
print('[OK] normalization identical to V2 Base')
PY
python scripts/inspect_routing_v2_checkpoint.py --checkpoint "${RUN}/final_model/pytorch_model.pt" --base "${BASE_CKPT}" --output "${RUN}/routing_v2_parameter_summary.json"
echo "${RUN}" > "${TASK_ROOT}/latest_skill_run.txt"
echo "[OK] T${TASK_ID} V2 skill: ${RUN}/final_model/pytorch_model.pt"
