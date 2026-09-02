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
BASE_RUN="${BASE_RUN:-$(cat "${V2_ROOT}/latest_base_run.txt")}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"
for file in "${BASE_CKPT}" "${BASE_STATS}"; do
  [ -f "${file}" ] || { echo "[ERROR] Missing Base artifact: ${file}"; exit 1; }
done

TASK_ROOT="${V2_ROOT}/task${TASK_ID}"
RUN_ROOT="${TASK_ROOT}/skill_runs"
LOG_ROOT="${TASK_ROOT}/logs"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-600}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
# Enforce final-only storage for the 2.5B policy checkpoint.
SAVE_INTERVAL=$((MAX_TRAIN_STEPS + 1))

ACTION_LR="${ACTION_LR:-0.0001}"
VLM_LR="${VLM_LR:-0.0001}"
VLM_LORA_RANK="${VLM_LORA_RANK:-32}"
VLM_LORA_ALPHA="${VLM_LORA_ALPHA:-32}"
COND_BOTTLENECK="${COND_BOTTLENECK:-128}"
NONLINEAR_BOTTLENECK="${NONLINEAR_BOTTLENECK:-128}"
DENSE_LAYERS="[12,13,14,15]"
NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"

if [ "${VLM_LORA_RANK}" -ne 32 ] || [ "${COND_BOTTLENECK}" -ne 128 ] || [ "${NONLINEAR_BOTTLENECK}" -ne 128 ]; then
  echo "[ERROR] Formal B2-only protocol requires Text-LoRA r32, Cond r128, NL r128."
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t${TASK_ID}_routing_v2_b2only_skill_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "======================================================================"
echo " Simplified Routing-V2 Skill / T${TASK_ID} -- FRESH from SAME Base"
echo " Base             : ${BASE_CKPT}"
echo " Trainable        : VLM Text-LoRA r32 + Action-B2"
echo " Action-B2        : Last4 Dense + Cond r128 + NL blocks 0-11 r128"
echo " Frozen/shared    : act query + flow query + QFormer + LaWM"
echo " Forbidden        : Query residuals + QFormer-LoRA + LaWM-LoRA + z*"
echo " Steps            : ${MAX_TRAIN_STEPS}; warmup=${NUM_WARMUP_STEPS}"
echo " Checkpoints      : final only"
echo " Expected params  : 90,583,040 (3.55% of 2.555B)"
echo "======================================================================"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((33400 + TASK_ID))}"

bash train_lawam.sh \
  "--run_root_dir=${RUN_ROOT}" \
  "--run_id=${RUN_ID}" \
  "--datasets.vla_data.cl_suite=libero_goal" \
  "--datasets.vla_data.cl_task_ids=[${TASK_ID}]" \
  "--datasets.vla_data.use_task_filtered_statistics=false" \
  "--trainer.use_pretrained_dataset_statistics=true" \
  "--trainer.pretrained_checkpoint=${BASE_CKPT}" \
  "--trainer.load_pretrained_policy_flow=true" \
  "--trainer.policy_flow_override_checkpoint=null" \
  "--framework.action_model.enable_expert_latent_aux=false" \
  "--framework.action_model.flow_cfg.enable_expert_latent_head=false" \
  "--framework.action_model.routing_v2_enable_query_delta=false" \
  "--framework.action_model.routing_v2_enable_memory=false" \
  "--framework.action_model.routing_v2_memory_train_mode=false" \
  "--framework.action_model.flow_cfg.residual_expert_num_blocks=0" \
  "--framework.action_model.flow_cfg.residual_expert_layer_indices=null" \
  "--framework.action_model.flow_cfg.conditioning_adapter_bottleneck=${COND_BOTTLENECK}" \
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
  "--trainer.freeze.train_flow_partial_dense=true" \
  "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYERS}" \
  "--trainer.freeze.train_flow_interface_dense=false" \
  "--trainer.freeze.train_flow_interface_lora=false" \
  "--trainer.freeze.train_vlm_flow_lora=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense=false" \
  "--trainer.freeze.train_vlm_text_lora_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_adapter=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_adapter=false" \
  "--trainer.freeze.train_flow_partial_dense_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_vlm_text_lora_partial_dense_conditioning_nonlinear_adapter=true" \
  "--trainer.freeze.train_flow_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_routing_v2_skill=false" \
  "--trainer.freeze.train_routing_v2_memory_only=false" \
  "--trainer.freeze.routing_v2_b2_only_skill_path=true" \
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

RUN=$(find "${RUN_ROOT}" -maxdepth 1 -type d -name "*+${RUN_ID}" | sort | tail -n 1)
[ -n "${RUN}" ] || { echo "[ERROR] Skill run not found"; exit 1; }
for file in "${RUN}/config.yaml" "${RUN}/dataset_statistics.json" "${RUN}/final_model/pytorch_model.pt"; do
  [ -f "${file}" ] || { echo "[ERROR] Missing ${file}"; exit 1; }
done

python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    base = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    task = json.load(stream)
for tag in base:
    for section in ("action", "state"):
        if base[tag][section] != task[tag][section]:
            raise RuntimeError(f"Normalization changed: {tag}/{section}")
print("[OK] action/state normalization is identical to the new Base")
PY

python scripts/audit_routing_v2_b2only_skill.py \
  --base "${BASE_CKPT}" \
  --skill "${RUN}/final_model/pytorch_model.pt" \
  --config "${RUN}/config.yaml" \
  --output "${RUN}/routing_v2_b2only_parameter_audit.json"

if find "${RUN}/checkpoints" -maxdepth 1 -type f -name '*_pytorch_model.pt' | rg -q .; then
  echo "[ERROR] Intermediate full policy checkpoints exist despite final-only protocol."
  exit 1
fi

echo "${RUN}" > "${TASK_ROOT}/latest_skill_run.txt"
echo "[OK] T${TASK_ID} simplified skill: ${RUN}/final_model/pytorch_model.pt"
