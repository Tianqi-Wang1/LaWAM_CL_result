#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LaWAM LIBERO-Goal T9 Residual Expert protocol
#
# Experiments (same Base upstream + same Base normalization):
#   C0: Base checkpoint -> FULL Flow fine-tune on T9 (protocol control)
#   Rb: Base upstream + frozen BASE Flow -> train only Residual Expert on T9
#   Rp: Base upstream + frozen PRETRAIN Flow -> train only Residual Expert on T9
#
# By default we use four residual blocks (last 4 DiT blocks). To sweep capacity:
#   RESIDUAL_BLOCKS="2 4" bash scripts/run_libero_goal_t9_residual_expert_protocol_v1.sh
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
VERIFY_SCRIPT="${ROOT}/scripts/verify_flow_residual_experiment.py"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${ROOT}/results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt}"

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_residual_expert_protocol_v1"
RUN_ROOT="${EXPERIMENT_ROOT}/runs"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/t9_residual_expert_protocol_v1"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
[ "${NUM_TRAIN_GPUS}" -gt 0 ] || { echo "[ERROR] TRAIN_GPUS is empty"; exit 1; }

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

# Which experiment families to execute.
RUN_FULL_FLOW_CONTROL="${RUN_FULL_FLOW_CONTROL:-true}"
RUN_RESIDUAL="${RUN_RESIDUAL:-true}"
FLOW_SOURCES="${FLOW_SOURCES:-base pretrain}"
RESIDUAL_BLOCKS="${RESIDUAL_BLOCKS:-4}"

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
unset NCCL_TOPO_FILE || true
unset NCCL_GRAPH_FILE || true
unset NCCL_CONF_FILE || true
unset HFAI_NCCL_OPT_LEVEL || true

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

verify_stats() {
    local reference="$1" current="$2"
    python - "${reference}" "${current}" <<'PY'
import json,sys
rp,cp=sys.argv[1:3]
with open(rp,"r",encoding="utf-8") as f:r=json.load(f)
with open(cp,"r",encoding="utf-8") as f:c=json.load(f)
for tag in r:
    for sec in ("action","state"):
        if r[tag][sec] != c[tag][sec]:
            raise RuntimeError(f"Normalization statistics changed: {tag}/{sec}")
print("[OK] action/state normalization is identical to Base.")
PY
}

if [ -n "${BASE_RUN:-}" ]; then :; else
    BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Formal Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

[ -f "${PRETRAIN_CKPT}" ] || { echo "[ERROR] Missing PRETRAIN_CKPT=${PRETRAIN_CKPT}"; exit 1; }
[ -f "${SUMMARY_SCRIPT}" ] || { echo "[ERROR] Missing ${SUMMARY_SCRIPT}"; exit 1; }
[ -f "${VERIFY_SCRIPT}" ] || { echo "[ERROR] Missing ${VERIFY_SCRIPT}"; exit 1; }

python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import LatentWorldPolicyFreezeConfig
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
ff=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
cf=set(ConditionalFlowMatchingConfig.__dataclass_fields__)
for key in ("train_flow_only","train_flow_lora","train_flow_residual_expert"):
    if key not in ff: raise RuntimeError(f"freeze_policy missing {key}")
for key in ("residual_expert_num_blocks","residual_expert_zero_init","residual_expert_scale"):
    if key not in cf: raise RuntimeError(f"flow config missing {key}")
print("[OK] Residual-Expert code support detected.")
PY

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/protocol_${PIPELINE_TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " LaWAM Goal T9 Residual Expert Protocol v1"
echo "=========================================================="
echo "Base upstream : ${BASE_CKPT}"
echo "Pretrain Flow : ${PRETRAIN_CKPT}"
echo "Task          : T9"
echo "Train GPUs    : ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "global batch  : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_GPUS))"
echo "steps / warmup: ${MAX_TRAIN_STEPS} / ${NUM_WARMUP_STEPS}"
echo "LR            : ${ACTION_LR}"
echo "Residual N    : ${RESIDUAL_BLOCKS}"
echo "Flow sources  : ${FLOW_SOURCES}"
echo "Trials        : ${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_TRAIN_GPUS}"

train_common_args=(
    "--run_root_dir=${RUN_ROOT}"
    "--datasets.vla_data.cl_suite=libero_goal"
    "--datasets.vla_data.cl_task_ids=[9]"
    "--datasets.vla_data.use_task_filtered_statistics=false"
    "--trainer.use_pretrained_dataset_statistics=true"
    "--trainer.pretrained_checkpoint=${BASE_CKPT}"
    "--trainer.load_pretrained_policy_flow=true"
    "--trainer.freeze.unfreeze_lam_decoder=false"
    "--trainer.learning_rate.vlm.lr=${ACTION_LR}"
    "--trainer.learning_rate.action_model.lr=${ACTION_LR}"
    "--trainer.learning_rate.world_model.lr=${ACTION_LR}"
    "--datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
    "--datasets.vla_data.num_workers=${NUM_WORKERS}"
    "--datasets.vla_data.val_num_workers=${VAL_NUM_WORKERS}"
    "--datasets.vla_data.persistent_workers=true"
    "--trainer.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    "--trainer.max_train_steps=${MAX_TRAIN_STEPS}"
    "--trainer.num_warmup_steps=${NUM_WARMUP_STEPS}"
    "--trainer.logging_frequency=${LOGGING_FREQUENCY}"
    "--trainer.eval_interval=${TRAIN_EVAL_INTERVAL}"
    "--trainer.eval_batches=${TRAIN_EVAL_BATCHES}"
    "--trainer.save_interval=${SAVE_INTERVAL}"
)

CONTROL_RUN=""
if [ "${RUN_FULL_FLOW_CONTROL}" = "true" ]; then
    RUN_ID="t9_full_flow_from_base_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
    echo ""
    echo "==================== C0: FULL FLOW FROM BASE ===================="
    bash train_lawam.sh \
        "${train_common_args[@]}" \
        "--run_id=${RUN_ID}" \
        "--trainer.policy_flow_override_checkpoint=null" \
        "--framework.action_model.flow_cfg.residual_expert_num_blocks=0" \
        "--trainer.freeze.train_flow_only=true" \
        "--trainer.freeze.train_flow_lora=false" \
        "--trainer.freeze.train_flow_residual_expert=false"

    CONTROL_RUN=$(find_run "${RUN_ROOT}" "*+${RUN_ID}")
    verify_run "C0 full-flow from Base" "${CONTROL_RUN}"
    verify_stats "${BASE_STATS}" "${CONTROL_RUN}/dataset_statistics.json"
    python "${VERIFY_SCRIPT}" \
        --mode full_flow \
        --base-ckpt "${BASE_CKPT}" \
        --final-ckpt "${CONTROL_RUN}/final_model/pytorch_model.pt"
fi

RESIDUAL_RUNS=()
if [ "${RUN_RESIDUAL}" = "true" ]; then
    for N_BLOCKS in ${RESIDUAL_BLOCKS}; do
        if ! [[ "${N_BLOCKS}" =~ ^[1-9][0-9]*$ ]]; then
            echo "[ERROR] Invalid residual block count: ${N_BLOCKS}"; exit 1
        fi
        for SOURCE in ${FLOW_SOURCES}; do
            case "${SOURCE}" in
                base)
                    FLOW_SOURCE_CKPT="${BASE_CKPT}"
                    FLOW_OVERRIDE="null"
                    ;;
                pretrain)
                    FLOW_SOURCE_CKPT="${PRETRAIN_CKPT}"
                    FLOW_OVERRIDE="${PRETRAIN_CKPT}"
                    ;;
                *)
                    echo "[ERROR] FLOW_SOURCES supports only: base pretrain; got ${SOURCE}"
                    exit 1
                    ;;
            esac

            RUN_ID="t9_residual_r${N_BLOCKS}_${SOURCE}flow_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
            echo ""
            echo "================ R${N_BLOCKS}: FROZEN ${SOURCE^^} FLOW + RESIDUAL ================"
            bash train_lawam.sh \
                "${train_common_args[@]}" \
                "--run_id=${RUN_ID}" \
                "--trainer.policy_flow_override_checkpoint=${FLOW_OVERRIDE}" \
                "--framework.action_model.flow_cfg.residual_expert_num_blocks=${N_BLOCKS}" \
                "--framework.action_model.flow_cfg.residual_expert_zero_init=true" \
                "--framework.action_model.flow_cfg.residual_expert_scale=1.0" \
                "--trainer.freeze.train_flow_only=false" \
                "--trainer.freeze.train_flow_lora=false" \
                "--trainer.freeze.train_flow_residual_expert=true"

            RUN=$(find_run "${RUN_ROOT}" "*+${RUN_ID}")
            verify_run "Residual R${N_BLOCKS} ${SOURCE}-Flow" "${RUN}"
            verify_stats "${BASE_STATS}" "${RUN}/dataset_statistics.json"

            python - "${RUN}/config.yaml" "${BASE_CKPT}" "${FLOW_OVERRIDE}" "${N_BLOCKS}" "${ACTION_LR}" <<'PY'
import math,sys
from omegaconf import OmegaConf
path,base,override,n_s,lr_s=sys.argv[1:6]
n=int(n_s); lr=float(lr_s)
cfg=OmegaConf.load(path); fr=cfg.trainer.freeze; fc=cfg.framework.action_model.flow_cfg
if str(cfg.trainer.pretrained_checkpoint) != str(base): raise RuntimeError("Residual run must use Base as main checkpoint")
if list(cfg.datasets.vla_data.cl_task_ids) != [9]: raise RuntimeError("Wrong task filter")
if not bool(cfg.trainer.load_pretrained_policy_flow): raise RuntimeError("Base Flow must first be loaded before optional override")
if bool(fr.get("train_flow_only",False)) or bool(fr.get("train_flow_lora",False)): raise RuntimeError("Wrong Flow training mode")
if not bool(fr.get("train_flow_residual_expert",False)): raise RuntimeError("Residual mode not enabled")
if bool(fr.get("unfreeze_lam_decoder",False)): raise RuntimeError("LAM decoder must stay frozen")
if int(fc.residual_expert_num_blocks) != n: raise RuntimeError("Wrong residual block count")
if not bool(fc.residual_expert_zero_init): raise RuntimeError("Residual expert must use zero-init")
actual_override=cfg.trainer.get("policy_flow_override_checkpoint",None)
if override == "null":
    if actual_override is not None: raise RuntimeError(f"Base-Flow run unexpectedly has override={actual_override}")
else:
    if str(actual_override) != str(override): raise RuntimeError("Pretrain-Flow override mismatch")
if not math.isclose(float(cfg.trainer.learning_rate.action_model.lr),lr,rel_tol=0,abs_tol=1e-12): raise RuntimeError("Unexpected action LR")
print("[OK] Residual training config verified.")
PY

            python "${VERIFY_SCRIPT}" \
                --mode residual \
                --base-ckpt "${BASE_CKPT}" \
                --flow-source-ckpt "${FLOW_SOURCE_CKPT}" \
                --final-ckpt "${RUN}/final_model/pytorch_model.pt" \
                --num-residual-blocks "${N_BLOCKS}"

            RESIDUAL_RUNS+=("${SOURCE}|${N_BLOCKS}|${RUN}")
        done
    done
fi

unset CUDA_VISIBLE_DEVICES || true
unset NUM_PROCESSES || true

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

run_eval() {
    local alias="$1" ckpt="$2"
    local eval_master="${OUTPUT_ROOT}/${PIPELINE_TIMESTAMP}/${alias}"
    mkdir -p "${eval_master}"
    SUITES="libero_goal" \
    TASK_IDS="9" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${EVAL_WORKERS}" \
    GPU_IDS="${POLICY_GPU}" \
    EVAL_GPU_IDS="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${eval_master}" \
    LIBERO_CKPT_ALIAS="${alias}" \
    bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${ckpt}"

    local eval_dir
    eval_dir=$(find "${eval_master}/${alias}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    [ -n "${eval_dir}" ] || { echo "[ERROR] Evaluation output missing for ${alias}"; exit 1; }
    local suite_dir="${eval_dir}/suites/libero_goal"
    python "${SUMMARY_SCRIPT}" --run-dir "${suite_dir}" --task-ids 9 --expected-trials "${NUM_TRIALS}"
    echo "[RESULT] ${alias}: ${suite_dir}/per_task_summary.csv"
}

if [ -n "${CONTROL_RUN}" ]; then
    run_eval "full_flow_from_base" "${CONTROL_RUN}/final_model/pytorch_model.pt"
fi
for item in "${RESIDUAL_RUNS[@]}"; do
    IFS='|' read -r source n_blocks run <<< "${item}"
    run_eval "residual_r${n_blocks}_${source}flow" "${run}/final_model/pytorch_model.pt"
done

echo "=========================================================="
echo " T9 Residual Expert protocol complete"
echo "=========================================================="
echo "Master log: ${MASTER_LOG}"
echo "Eval root : ${OUTPUT_ROOT}/${PIPELINE_TIMESTAMP}"
if [ -n "${CONTROL_RUN}" ]; then echo "Control   : ${CONTROL_RUN}"; fi
for item in "${RESIDUAL_RUNS[@]}"; do echo "Residual  : ${item}"; done
echo "=========================================================="
