#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# E3: T9 R4-LAST + distributed Flow LoRA r=8
# Base upstream + frozen Base Flow.
# Trainable:
#   (1) R4 nonlinear residual experts on final DiT layers [12,13,14,15]
#   (2) LoRA r=8 on ORIGINAL 16 DiT blocks only:
#       attn1 Q/K/V/O + FFN in/out (96 target Linear modules)
# Frozen:
#   enc_vlm, output projections, action encoder/decoder, all original weights,
#   and the entire upstream VLM/LaWM/QFormer stack.
# After training, LoRA is merged into dense Base Flow weights for evaluation;
# residual experts remain explicit in the architecture.
# Train GPUs: 4,5,6,7. Eval: policy GPU 4 + LIBERO GPU 5.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_residual_followup_v2/r4_plus_lora_r8"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_residual_followup_v2/r4_plus_lora_r8"
VERIFY_SCRIPT="${ROOT}/scripts/verify_flow_residual_v2.py"
MERGE_SCRIPT="${ROOT}/scripts/merge_flow_lora_checkpoint.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
ACTION_LR="${ACTION_LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
EXPECTED_LORA_TARGETS=96

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE NCCL_GRAPH_FILE NCCL_CONF_FILE HFAI_NCCL_OPT_LEVEL 2>/dev/null || true

find_run() {
    local root="$1" pattern="$2"
    find "${root}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}
verify_run() {
    local label="$1" run="$2"
    [ -n "${run}" ] || { echo "[ERROR] ${label}: run not found"; exit 1; }
    for f in "${run}/config.yaml" "${run}/dataset_statistics.json" "${run}/final_model/pytorch_model.pt"; do
        [ -f "${f}" ] || { echo "[ERROR] ${label}: missing ${f}"; exit 1; }
    done
    echo "[OK] ${label}: ${run}"
}

if [ -z "${BASE_RUN:-}" ]; then
    BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Formal Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

python - <<'PY'
import inspect
from starVLA.model.framework.latent_world.runtime.freeze_policy import LatentWorldPolicyFreezeConfig
from starVLA.model.framework.latent_world.runtime.flow_lora import inject_flow_conditioning_lora
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
ff=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
cf=set(ConditionalFlowMatchingConfig.__dataclass_fields__)
for key in ("train_flow_lora","train_flow_residual_expert","flow_lora_target_attention","flow_lora_target_ffn"):
    if key not in ff: raise RuntimeError(f"Missing freeze support: {key}")
if "residual_expert_layer_indices" not in cf: raise RuntimeError("Missing residual placement v2 support")
sig=inspect.signature(inject_flow_conditioning_lora)
for key in ("target_attention","target_ffn","target_enc_vlm","target_output"):
    if key not in sig.parameters: raise RuntimeError(f"Generalized LoRA injector missing {key}")
print("[OK] Residual+LoRA hybrid code support detected.")
PY

[ -f "${MERGE_SCRIPT}" ] || { echo "[ERROR] Missing ${MERGE_SCRIPT}"; exit 1; }

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t9_r4_last_plus_lora_r${LORA_RANK}_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " T9 R4-LAST + distributed LoRA r=${LORA_RANK}"
echo "=========================================================="
echo "Base          : ${BASE_CKPT}"
echo "Train GPUs    : ${TRAIN_GPUS}"
echo "Global batch  : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Steps/warmup  : ${MAX_TRAIN_STEPS}/${NUM_WARMUP_STEPS}"
echo "LR            : ${ACTION_LR}"
echo "Residual      : final 4 layers [12..15]"
echo "LoRA          : rank=${LORA_RANK}, alpha=${LORA_ALPHA}, targets=${EXPECTED_LORA_TARGETS} original DiT linears"
echo "LoRA excludes : enc_vlm, output, AdaNorm, timestep, residual blocks"
echo "Eval GPUs     : policy=${POLICY_GPU}, libero=${EVAL_GPU}"
echo "Trials        : ${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29833}"

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
    "--framework.action_model.flow_cfg.residual_expert_num_blocks=4" \
    "--framework.action_model.flow_cfg.residual_expert_layer_indices=null" \
    "--framework.action_model.flow_cfg.residual_expert_zero_init=true" \
    "--framework.action_model.flow_cfg.residual_expert_scale=1.0" \
    "--trainer.freeze.train_flow_only=false" \
    "--trainer.freeze.train_flow_lora=true" \
    "--trainer.freeze.train_flow_residual_expert=true" \
    "--trainer.freeze.flow_lora_rank=${LORA_RANK}" \
    "--trainer.freeze.flow_lora_alpha=${LORA_ALPHA}" \
    "--trainer.freeze.flow_lora_dropout=${LORA_DROPOUT}" \
    "--trainer.freeze.flow_lora_target_attention=true" \
    "--trainer.freeze.flow_lora_target_ffn=true" \
    "--trainer.freeze.flow_lora_target_enc_vlm=false" \
    "--trainer.freeze.flow_lora_target_output=false" \
    "--trainer.freeze.flow_lora_target_adanorm=false" \
    "--trainer.freeze.flow_lora_target_timestep=false" \
    "--trainer.freeze.unfreeze_lam_decoder=false" \
    "--trainer.learning_rate.vlm.lr=${ACTION_LR}" \
    "--trainer.learning_rate.action_model.lr=${ACTION_LR}" \
    "--trainer.learning_rate.world_model.lr=${ACTION_LR}" \
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
verify_run "R4+LoRA" "${RUN}"
UNMERGED="${RUN}/final_model/pytorch_model.pt"
MERGED="${RUN}/final_model/pytorch_model_merged.pt"
ADAPTER="${RUN}/final_model/flow_lora_adapter.pt"

python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json,sys
with open(sys.argv[1],"r",encoding="utf-8") as f:a=json.load(f)
with open(sys.argv[2],"r",encoding="utf-8") as f:b=json.load(f)
for tag in a:
    for sec in ("action","state"):
        if a[tag][sec] != b[tag][sec]: raise RuntimeError(f"Normalization changed: {tag}/{sec}")
print("[OK] action/state normalization identical to Base.")
PY

python - "${RUN}/config.yaml" "${BASE_CKPT}" "${LORA_RANK}" "${LORA_ALPHA}" <<'PY'
import math,sys
from omegaconf import OmegaConf
cfg=OmegaConf.load(sys.argv[1]); base=sys.argv[2]; rank=int(sys.argv[3]); alpha=float(sys.argv[4])
fr=cfg.trainer.freeze; fc=cfg.framework.action_model.flow_cfg
if str(cfg.trainer.pretrained_checkpoint) != base: raise RuntimeError("Wrong Base checkpoint")
if list(cfg.datasets.vla_data.cl_task_ids) != [9]: raise RuntimeError("Wrong task filter")
if bool(fr.train_flow_only) or not bool(fr.train_flow_lora) or not bool(fr.train_flow_residual_expert): raise RuntimeError("Hybrid mode not enabled")
if int(fc.residual_expert_num_blocks) != 4 or fc.residual_expert_layer_indices is not None: raise RuntimeError("Hybrid must use final-4 residual placement")
if int(fr.flow_lora_rank) != rank or not math.isclose(float(fr.flow_lora_alpha),alpha): raise RuntimeError("Wrong LoRA rank/alpha")
expected={
    "flow_lora_target_attention":True,
    "flow_lora_target_ffn":True,
    "flow_lora_target_enc_vlm":False,
    "flow_lora_target_output":False,
    "flow_lora_target_adanorm":False,
    "flow_lora_target_timestep":False,
}
for k,v in expected.items():
    if bool(fr.get(k)) != v: raise RuntimeError(f"{k}={fr.get(k)} != {v}")
print("[OK] R4+LoRA config verified.")
PY

python "${VERIFY_SCRIPT}" \
    --mode hybrid \
    --base-ckpt "${BASE_CKPT}" \
    --final-ckpt "${UNMERGED}" \
    --num-residual-blocks 4 \
    --expected-lora-targets "${EXPECTED_LORA_TARGETS}"

# Merge LoRA into the original dense Flow linears for standard inference.
# Residual expert tensors are left untouched in the checkpoint.
python "${MERGE_SCRIPT}" \
    "${UNMERGED}" \
    --output "${MERGED}" \
    --adapter-output "${ADAPTER}" \
    --alpha "${LORA_ALPHA}"

[ -f "${MERGED}" ] || { echo "[ERROR] Missing merged checkpoint ${MERGED}"; exit 1; }
python - "${MERGED}" <<'PY'
import sys,torch
p=sys.argv[1]
try: s=torch.load(p,map_location="cpu",weights_only=True,mmap=True)
except TypeError: s=torch.load(p,map_location="cpu")
left=[k for k in s if ".lora_" in k]
if left: raise RuntimeError(f"Merged checkpoint still contains LoRA keys: {left[:20]}")
res=[k for k in s if "DiT.residual_expert_blocks." in k]
if not res: raise RuntimeError("Merged checkpoint lost residual expert tensors")
print(f"[OK] merged checkpoint: no LoRA keys; residual tensors={len(res)}")
PY

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

ALIAS="t9_r4_last_plus_lora_r${LORA_RANK}"
EVAL_MASTER="${OUTPUT_ROOT}/${STAMP}/${ALIAS}"
mkdir -p "${EVAL_MASTER}"
SUITES="libero_goal" \
TASK_IDS="9" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
NUM_WORKERS="${EVAL_WORKERS}" \
GPU_IDS="${POLICY_GPU}" \
EVAL_GPU_IDS="${EVAL_GPU}" \
SAVE_VIDEOS="${SAVE_VIDEOS}" \
OUTPUT_ROOT="${EVAL_MASTER}" \
LIBERO_CKPT_ALIAS="${ALIAS}" \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh \
    "${MERGED}"

EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
[ -n "${EVAL_DIR}" ] || { echo "[ERROR] Evaluation output missing"; exit 1; }
SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
python "${SUMMARY_SCRIPT}" --run-dir "${SUITE_DIR}" --task-ids 9 --expected-trials "${NUM_TRIALS}"

echo "=========================================================="
echo " R4 + LoRA COMPLETE"
echo " Run       : ${RUN}"
echo " Unmerged  : ${UNMERGED}"
echo " Merged    : ${MERGED}"
echo " Adapter   : ${ADAPTER}"
echo " Summary   : ${SUITE_DIR}/per_task_summary.csv"
echo " Master log: ${MASTER_LOG}"
echo "=========================================================="
