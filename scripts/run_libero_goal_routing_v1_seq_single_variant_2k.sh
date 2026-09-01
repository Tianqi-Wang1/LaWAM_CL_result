#!/usr/bin/env bash
set -euo pipefail

# Fresh single-task Routing-V1 experiment from the NEW latent-enabled Base.
# Evaluation still uses the provided task ID (oracle expert selection is not yet changed).

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam
ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

MODE="${MODE:-b1}"
TASK_ID="${TASK_ID:-9}"
case "${MODE}" in b1|b2) ;; *) echo "[ERROR] MODE must be b1 or b2"; exit 2 ;; esac
case "${TASK_ID}" in 6|7|8|9) ;; *) echo "[ERROR] TASK_ID must be 6/7/8/9"; exit 2 ;; esac
STAGE=$((TASK_ID - 5))

ROUTING_ROOT="${ROUTING_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/routing_v1_base10k_expert2k}"
if [ -z "${BASE_RUN:-}" ]; then
  if [ -f "${ROUTING_ROOT}/latest_base_run.txt" ]; then BASE_RUN=$(cat "${ROUTING_ROOT}/latest_base_run.txt"); fi
fi
if [ -z "${BASE_RUN:-}" ]; then
  BASE_RUN=$(find "${ROUTING_ROOT}/base_runs" -maxdepth 1 -type d -name '*+routing_v1_base_t0_5_*' | sort | tail -n 1)
fi
[ -n "${BASE_RUN}" ] || { echo "[ERROR] Routing-V1 Base not found. Run run_libero_goal_routing_v1_base_10k.sh first."; exit 1; }
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"
[ -f "${BASE_CKPT}" ] && [ -f "${BASE_STATS}" ] || { echo "[ERROR] Invalid BASE_RUN=${BASE_RUN}"; exit 1; }

ACTION_LR="${ACTION_LR:-0.0001}"
VLM_LR="${VLM_LR:-0.0001}"
CONDITIONING_BOTTLENECK="${CONDITIONING_BOTTLENECK:-128}"
NONLINEAR_BOTTLENECK="${NONLINEAR_BOTTLENECK:-128}"
LATENT_HEAD_HIDDEN="${LATENT_HEAD_HIDDEN:-1024}"
LATENT_WEIGHT="${LATENT_WEIGHT:-0.1}"
WORLD_WEIGHT="${WORLD_WEIGHT:-0.1}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"; NUM_GPUS="${#GPU_ARRAY[@]}"
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
DO_EVAL="${DO_EVAL:-false}"
DO_MERGE="${DO_MERGE:-false}"

DENSE_LAYERS="[12,13,14,15]"
NONLINEAR_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11]"
TRAIN_PD=true
TRAIN_PD_COND_NL=false
TRAIN_TEXT_PD_COND_NL=false
HAS_LORA=false
VLM_LORA_RANK=32
VLM_LORA_ALPHA=32
if [ "${MODE}" = "b1" ]; then
  LABEL="B1 + Routing-V1 latent head"
  RUN_TAG="b1_latent_v1_last4_cond128_nl0_11_128"
  TRAIN_PD_COND_NL=true
  MODE_OFFSET=100
else
  LABEL="B2 + Routing-V1 latent head"
  RUN_TAG="b2_latent_v1_last4_cond128_nl0_11_128_textlora32"
  TRAIN_TEXT_PD_COND_NL=true
  HAS_LORA=true
  MODE_OFFSET=200
fi

export TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1 STARVLA_WORKER_OMP_THREADS=1 OMP_NUM_THREADS=1
export WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

EXPERIMENT_ROOT="${ROUTING_ROOT}/${MODE}/task${TASK_ID}"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"; LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/routing_v1_base10k_expert2k/${MODE}/task${TASK_ID}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t${TASK_ID}_${RUN_TAG}_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " ${LABEL} / T${TASK_ID}"
echo " Fresh init       : ${BASE_CKPT}"
echo " Shared VLM       : frozen (B2 only adds Text-LoRA-r32)"
echo " act/flow queries : frozen"
echo " VLM->LAM QFormer : frozen"
echo " LaWM             : frozen (gradient may pass through it to z*)"
echo " Dense DiT        : ${DENSE_LAYERS}"
echo " Nonlinear DiT    : ${NONLINEAR_LAYERS}"
echo " z* head          : trainable, hidden=${LATENT_HEAD_HIDDEN}"
echo " Aux weights      : z=${LATENT_WEIGHT}, world=${WORLD_WEIGHT}"
echo " Steps            : ${MAX_TRAIN_STEPS}"
echo " Evaluation       : current task ID T${TASK_ID}; no routing yet"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((32000 + MODE_OFFSET + TASK_ID))}"

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
  "--framework.action_model.enable_expert_latent_aux=true" \
  "--framework.action_model.expert_latent_distill_weight=${LATENT_WEIGHT}" \
  "--framework.action_model.expert_world_loss_weight=${WORLD_WEIGHT}" \
  "--framework.action_model.expert_latent_loss_type=mse" \
  "--framework.action_model.flow_cfg.enable_expert_latent_head=true" \
  "--framework.action_model.flow_cfg.expert_latent_dim=-1" \
  "--framework.action_model.flow_cfg.expert_latent_head_hidden_dim=${LATENT_HEAD_HIDDEN}" \
  "--framework.action_model.flow_cfg.expert_latent_head_dropout=0.0" \
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
  "--trainer.freeze.train_flow_conditioning_nonlinear_adapter=false" \
  "--trainer.freeze.train_expert_latent_head=true" \
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
[ -n "${RUN}" ] || { echo "[ERROR] Run not found"; exit 1; }
for f in "${RUN}/config.yaml" "${RUN}/dataset_statistics.json" "${RUN}/final_model/pytorch_model.pt"; do [ -f "${f}" ] || { echo "[ERROR] Missing ${f}"; exit 1; }; done
UNMERGED="${RUN}/final_model/pytorch_model.pt"
MERGED="${RUN}/final_model/pytorch_model_merged.pt"

python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f:a=json.load(f)
with open(sys.argv[2],encoding='utf-8') as f:b=json.load(f)
for tag in a:
    for sec in ('action','state'):
        if a[tag][sec] != b[tag][sec]: raise RuntimeError(f'Normalization changed: {tag}/{sec}')
print('[OK] action/state normalization identical to new Routing-V1 Base.')
PY

python scripts/verify_routing_v1_checkpoint.py \
  --checkpoint "${UNMERGED}" --config "${RUN}/config.yaml" --base "${BASE_CKPT}" --mode "${MODE}"

if [ "${HAS_LORA}" = true ]; then
  if [ "${DO_MERGE,,}" = "true" ]; then
    python scripts/merge_lora_checkpoint_v6.py --input "${UNMERGED}" --output "${MERGED}" --alpha "${VLM_LORA_ALPHA}"
    EVAL_CKPT="${MERGED}"
  else
    EVAL_CKPT="${UNMERGED}"
    echo "[INFO] DO_MERGE=false: keeping B2 checkpoint unmerged for now."
    echo "[INFO] The smoke-chain final evaluation will merge it just-in-time."
  fi
else
  EVAL_CKPT="${UNMERGED}"
fi

if [ "${DO_EVAL,,}" = "true" ]; then
  if [ "${HAS_LORA}" = true ] && [ "${DO_MERGE,,}" != "true" ]; then
    echo "[ERROR] B2 closed-loop evaluation requires a merged checkpoint. Set DO_MERGE=true or DO_EVAL=false."
    exit 1
  fi
  export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
  export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
  export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
  ALIAS="t${TASK_ID}_${RUN_TAG}"
  EVAL_MASTER="${OUTPUT_ROOT}/${STAMP}/${ALIAS}"; mkdir -p "${EVAL_MASTER}"
  SUITES="libero_goal" TASK_IDS="${TASK_ID}" NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
  NUM_WORKERS="${EVAL_WORKERS}" GPU_IDS="${POLICY_GPU}" EVAL_GPU_IDS="${EVAL_GPU}" \
  SAVE_VIDEOS="${SAVE_VIDEOS}" OUTPUT_ROOT="${EVAL_MASTER}" LIBERO_CKPT_ALIAS="${ALIAS}" \
  bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${EVAL_CKPT}"
  EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
  SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
  python scripts/summarize_libero_cl_eval.py --run-dir "${SUITE_DIR}" --task-ids "${TASK_ID}" --expected-trials "${NUM_TRIALS}"
  echo "${SUITE_DIR}/per_task_summary.csv" > "${EXPERIMENT_ROOT}/latest_summary_path.txt"
fi

cat > "${EXPERIMENT_ROOT}/PROTOCOL.txt" <<EOF
Routing-V1 ${MODE^^} / T${TASK_ID}
Initialization: fresh from NEW Routing-V1 Base ${BASE_CKPT}
Training: ${MAX_TRAIN_STEPS} steps
Shared VLM: frozen; B2 adds text-only LoRA-r32
act_query / flow_action_query / VLM-to-LAM QFormer: frozen
LaWM decoder: frozen
Action Expert: B1/B2 protocol + trainable z* head
Auxiliary targets: z_GT from frozen IDM and h_GT from real future DINO feature
Evaluation: provided task ID T${TASK_ID}; routing is NOT enabled in this experiment
EOF

echo "=========================================================="
echo " COMPLETE ${MODE^^} Routing-V1 / T${TASK_ID}"
echo " Run       : ${RUN}"
echo " Eval ckpt : ${EVAL_CKPT}"
echo " Log       : ${MASTER_LOG}"
echo "=========================================================="
