#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# T9 Joint VLM + Action-Flow LoRA location ablation (single variant)
#
# FIXED action-side LoRA for every variant:
#   all 16 original Flow DiT blocks
#   Attention Q/K/V/O + FFN input/output only
#   NO enc_vlm / proj_out / action encoder-decoder / timestep / residual expert
#
# VLM_VARIANT controls ONLY VLM LoRA location:
#   text_last4 | text_last8 | text_all | vision | merger | text_vision | full
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

VLM_VARIANT="${VLM_VARIANT:-text_all}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
FLOW_LORA_LR="${FLOW_LORA_LR:-0.0001}"
VLM_LORA_LR="${VLM_LORA_LR:-0.0001}"

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

case "${VLM_VARIANT}" in
  text_last4)
    VLM_TEXT=true;  VLM_TEXT_LAST_N=4; VLM_VISION=false; VLM_MERGER=false ;;
  text_last8)
    VLM_TEXT=true;  VLM_TEXT_LAST_N=8; VLM_VISION=false; VLM_MERGER=false ;;
  text_all)
    VLM_TEXT=true;  VLM_TEXT_LAST_N=0; VLM_VISION=false; VLM_MERGER=false ;;
  vision)
    VLM_TEXT=false; VLM_TEXT_LAST_N=0; VLM_VISION=true;  VLM_MERGER=false ;;
  merger)
    VLM_TEXT=false; VLM_TEXT_LAST_N=0; VLM_VISION=false; VLM_MERGER=true ;;
  text_vision)
    VLM_TEXT=true;  VLM_TEXT_LAST_N=0; VLM_VISION=true;  VLM_MERGER=false ;;
  full)
    VLM_TEXT=true;  VLM_TEXT_LAST_N=0; VLM_VISION=true;  VLM_MERGER=true ;;
  *)
    echo "[ERROR] Unknown VLM_VARIANT=${VLM_VARIANT}"
    echo "        valid: text_last4 text_last8 text_all vision merger text_vision full"
    exit 2 ;;
esac

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
from starVLA.model.framework.latent_world.runtime.vlm_lora import inject_vlm_lora
fields=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required={
 'train_vlm_flow_lora','vlm_lora_target_text','vlm_lora_text_last_n',
 'vlm_lora_target_vision','vlm_lora_target_merger','vlm_lora_rank',
}
missing=sorted(required-fields)
if missing: raise RuntimeError(f'Missing v5 joint VLM+Flow LoRA fields: {missing}')
print('[OK] Joint VLM+Flow LoRA v5 support detected.')
PYCODE

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_vlm_flow_lora_v5/${VLM_VARIANT}"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_vlm_flow_lora_v5/${VLM_VARIANT}"
MERGE_SCRIPT="${ROOT}/scripts/merge_vlm_flow_lora_checkpoint_v5.py"
VERIFY_SCRIPT="${ROOT}/scripts/verify_vlm_flow_lora_checkpoint_v5.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t9_vlm_${VLM_VARIANT}_flow_dit16_lora_r${LORA_RANK}_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " T9 VLM + ACTION-FLOW LoRA-r${LORA_RANK}: ${VLM_VARIANT}"
echo "=========================================================="
echo "Base          : ${BASE_CKPT}"
echo "Train GPUs    : ${TRAIN_GPUS}"
echo "Global batch  : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Steps/warmup  : ${MAX_TRAIN_STEPS}/${NUM_WARMUP_STEPS}"
echo "Flow LR       : ${FLOW_LORA_LR}"
echo "VLM LR        : ${VLM_LORA_LR}"
echo "LoRA r/alpha  : ${LORA_RANK}/${LORA_ALPHA}"
echo "Action target : DiT 0-15 Q/K/V/O + FFN ONLY"
echo "VLM text      : ${VLM_TEXT}; last_n=${VLM_TEXT_LAST_N} (0=all retained 16)"
echo "VLM vision    : ${VLM_VISION}; visual transformer blocks only"
echo "VLM merger    : ${VLM_MERGER}; main + DeepStack mergers"
echo "Eval GPUs     : policy=${POLICY_GPU}, libero=${EVAL_GPU}"
echo "Trials        : ${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-30131}"

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
    "--trainer.freeze.freeze_vlm_all=true" \
    "--trainer.freeze.freeze_act_query=true" \
    "--trainer.freeze.freeze_flow_action_query=true" \
    "--trainer.freeze.train_flow_only=false" \
    "--trainer.freeze.train_flow_lora=false" \
    "--trainer.freeze.train_flow_residual_expert=false" \
    "--trainer.freeze.train_flow_partial_dense=false" \
    "--trainer.freeze.train_flow_interface_dense=false" \
    "--trainer.freeze.train_flow_interface_lora=false" \
    "--trainer.freeze.train_vlm_flow_lora=true" \
    "--trainer.freeze.vlm_lora_target_text=${VLM_TEXT}" \
    "--trainer.freeze.vlm_lora_text_last_n=${VLM_TEXT_LAST_N}" \
    "--trainer.freeze.vlm_lora_target_vision=${VLM_VISION}" \
    "--trainer.freeze.vlm_lora_target_merger=${VLM_MERGER}" \
    "--trainer.freeze.vlm_lora_rank=${LORA_RANK}" \
    "--trainer.freeze.vlm_lora_alpha=${LORA_ALPHA}" \
    "--trainer.freeze.vlm_lora_dropout=${LORA_DROPOUT}" \
    "--trainer.freeze.flow_lora_rank=${LORA_RANK}" \
    "--trainer.freeze.flow_lora_alpha=${LORA_ALPHA}" \
    "--trainer.freeze.flow_lora_dropout=${LORA_DROPOUT}" \
    "--trainer.freeze.unfreeze_lam_decoder=false" \
    "--trainer.learning_rate.action_model.lr=${FLOW_LORA_LR}" \
    "--trainer.learning_rate.vlm.lr=${VLM_LORA_LR}" \
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
verify_run "T9 joint LoRA ${VLM_VARIANT}" "${RUN}"
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

python - "${RUN}/config.yaml" "${BASE_CKPT}" "${VLM_VARIANT}" "${LORA_RANK}" "${LORA_ALPHA}" \
  "${VLM_TEXT}" "${VLM_TEXT_LAST_N}" "${VLM_VISION}" "${VLM_MERGER}" <<'PYCODE'
import math,sys
from omegaconf import OmegaConf
path,base,variant,rank_s,alpha_s,text_s,lastn_s,vision_s,merger_s=sys.argv[1:10]
def b(x): return str(x).lower()=='true'
cfg=OmegaConf.load(path); fr=cfg.trainer.freeze
if str(cfg.trainer.pretrained_checkpoint) != base: raise RuntimeError('Wrong Base checkpoint')
if list(cfg.datasets.vla_data.cl_task_ids) != [9]: raise RuntimeError('Wrong task filter')
if not bool(fr.train_vlm_flow_lora): raise RuntimeError('Joint VLM+Flow LoRA mode is off')
for k in ('train_flow_only','train_flow_lora','train_flow_residual_expert','train_flow_partial_dense','train_flow_interface_dense','train_flow_interface_lora'):
    if bool(fr.get(k,False)): raise RuntimeError(f'Unexpected mode enabled: {k}')
checks={
 'vlm_lora_target_text':b(text_s), 'vlm_lora_text_last_n':int(lastn_s),
 'vlm_lora_target_vision':b(vision_s), 'vlm_lora_target_merger':b(merger_s),
 'vlm_lora_rank':int(rank_s), 'flow_lora_rank':int(rank_s),
}
for k,v in checks.items():
    if fr.get(k,None) != v: raise RuntimeError(f'{k}: {fr.get(k,None)} != {v}')
if not math.isclose(float(fr.vlm_lora_alpha),float(alpha_s),abs_tol=1e-12): raise RuntimeError('Wrong VLM alpha')
if not math.isclose(float(fr.flow_lora_alpha),float(alpha_s),abs_tol=1e-12): raise RuntimeError('Wrong Flow alpha')
print(f'[OK] v5 joint LoRA config verified: {variant}')
PYCODE

python "${MERGE_SCRIPT}" --input "${UNMERGED}" --output "${MERGED}" --alpha "${LORA_ALPHA}"
python "${VERIFY_SCRIPT}" --base "${BASE_CKPT}" --unmerged "${UNMERGED}" --merged "${MERGED}" \
  --variant "${VLM_VARIANT}" --rank "${LORA_RANK}"

if [ "${DO_EVAL,,}" = "true" ]; then
  export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
  export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
  export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
  ALIAS="t9_vlm_${VLM_VARIANT}_flow_dit16_lora_r${LORA_RANK}"
  EVAL_MASTER="${OUTPUT_ROOT}/${STAMP}/${ALIAS}"
  mkdir -p "${EVAL_MASTER}"
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
echo " T9 VLM+FLOW LoRA ${VLM_VARIANT} COMPLETE"
echo " Run       : ${RUN}"
echo " Merged    : ${MERGED}"
echo " Summary   : ${SUITE_DIR}/per_task_summary.csv"
echo " Master log: ${MASTER_LOG}"
echo "=========================================================="
