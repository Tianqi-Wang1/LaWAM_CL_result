#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# T9 FT-LAST4: dense fine-tune original Base Flow layers 12,13,14,15
# Only the selected ORIGINAL Base Flow DiT blocks are trainable at full rank.
# No residual expert, no LoRA, no upstream update.
# Train GPUs: 4,5,6,7. Eval: policy GPU 4 + LIBERO GPU 5.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_partial_dense_followup_v3/ft_last4"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_partial_dense_followup_v3/ft_last4"
VERIFY_SCRIPT="${ROOT}/scripts/verify_flow_partial_dense_v3.py"
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
DENSE_LAYER_INDICES="[12,13,14,15]"
DENSE_LAYER_CSV="12,13,14,15"

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

python - <<'PYCODE'
from starVLA.model.framework.latent_world.runtime.freeze_policy import LatentWorldPolicyFreezeConfig
ff=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
for key in ("train_flow_partial_dense", "flow_partial_dense_layer_indices"):
    if key not in ff:
        raise RuntimeError(f"Missing partial-dense Flow support: {key}")
print("[OK] Partial-dense Flow code support detected.")
PYCODE

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="t9_ft_last4_${MAX_TRAIN_STEPS}step_4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
MASTER_LOG="${LOG_ROOT}/${RUN_ID}_${STAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " T9 FT-LAST4: dense fine-tune original Base Flow layers 12,13,14,15"
echo "=========================================================="
echo "Base          : ${BASE_CKPT}"
echo "Train GPUs    : ${TRAIN_GPUS}"
echo "Global batch  : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo "Steps/warmup  : ${MAX_TRAIN_STEPS}/${NUM_WARMUP_STEPS}"
echo "LR            : ${ACTION_LR}"
echo "Dense layers  : ${DENSE_LAYER_INDICES}"
echo "Adaptation    : ORIGINAL Base Flow blocks, full-rank in-place FT"
echo "Residual/LoRA : disabled / disabled"
echo "Eval GPUs     : policy=${POLICY_GPU}, libero=${EVAL_GPU}"
echo "Trials        : ${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_GPUS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29931}"

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
    "--trainer.freeze.train_flow_only=false" \
    "--trainer.freeze.train_flow_lora=false" \
    "--trainer.freeze.train_flow_residual_expert=false" \
    "--trainer.freeze.train_flow_partial_dense=true" \
    "--trainer.freeze.flow_partial_dense_layer_indices=${DENSE_LAYER_INDICES}" \
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
verify_run "FT-LAST4" "${RUN}"

python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PYCODE'
import json,sys
with open(sys.argv[1],"r",encoding="utf-8") as f:a=json.load(f)
with open(sys.argv[2],"r",encoding="utf-8") as f:b=json.load(f)
for tag in a:
    for sec in ("action","state"):
        if a[tag][sec] != b[tag][sec]:
            raise RuntimeError(f"Normalization changed: {tag}/{sec}")
print("[OK] action/state normalization identical to Base.")
PYCODE

python - "${RUN}/config.yaml" "${BASE_CKPT}" <<'PYCODE'
import sys
from omegaconf import OmegaConf
cfg=OmegaConf.load(sys.argv[1]); base=sys.argv[2]
fr=cfg.trainer.freeze; fc=cfg.framework.action_model.flow_cfg
expected=[12, 13, 14, 15]
if str(cfg.trainer.pretrained_checkpoint) != base: raise RuntimeError("Wrong Base checkpoint")
if list(cfg.datasets.vla_data.cl_task_ids) != [9]: raise RuntimeError("Wrong task filter")
if bool(fr.train_flow_only) or bool(fr.train_flow_lora) or bool(fr.train_flow_residual_expert):
    raise RuntimeError("A different Flow adaptation mode leaked into partial-dense FT")
if not bool(fr.train_flow_partial_dense): raise RuntimeError("Partial-dense FT is not enabled")
if list(fr.flow_partial_dense_layer_indices) != expected:
    raise RuntimeError(f"Wrong dense layers: {fr.flow_partial_dense_layer_indices} vs {expected}")
if int(fc.residual_expert_num_blocks) != 0: raise RuntimeError("Residual experts must be disabled")
if fc.residual_expert_layer_indices is not None: raise RuntimeError("Residual layer indices must be null")
print("[OK] Partial-dense config verified.")
PYCODE

python "${VERIFY_SCRIPT}" \
    --base-ckpt "${BASE_CKPT}" \
    --final-ckpt "${RUN}/final_model/pytorch_model.pt" \
    --layers "${DENSE_LAYER_CSV}"

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

ALIAS="t9_ft_last4"
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
    "${RUN}/final_model/pytorch_model.pt"

EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
[ -n "${EVAL_DIR}" ] || { echo "[ERROR] Evaluation output missing"; exit 1; }
SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
python "${SUMMARY_SCRIPT}" --run-dir "${SUITE_DIR}" --task-ids 9 --expected-trials "${NUM_TRIALS}"

echo "=========================================================="
echo " FT-LAST4 COMPLETE"
echo " Run       : ${RUN}"
echo " Summary   : ${SUITE_DIR}/per_task_summary.csv"
echo " Master log: ${MASTER_LOG}"
echo "=========================================================="
