#!/usr/bin/env bash
set -euo pipefail

# Control experiment for the Fresh Strict Full-Flow T9 result.
#
# The pretrained and scratch Full-Flow experiments must differ in exactly one
# setting:
#   pretrained: trainer.load_pretrained_policy_flow=true
#   scratch:    trainer.load_pretrained_policy_flow=false
#
# Formal Base is still used to initialize every frozen upstream tensor and to
# provide the original action/state normalization statistics.

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/full_flow_t9_scratch_control"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/full_flow_t9_scratch_control"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
[ "${NUM_TRAIN_GPUS}" -gt 0 ] || { echo "[ERROR] TRAIN_GPUS is empty"; exit 1; }

# These defaults intentionally match run_libero_goal_full_flow_t9_retrain_v1.sh.
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
SEED="${SEED:-2026}"
REPEATED_DIFFUSION_STEPS="${REPEATED_DIFFUSION_STEPS:-2}"

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

if [ -n "${BASE_RUN:-}" ]; then :; else
    BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Formal Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

python - <<'PY'
from inspect import getsource

from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

fields = set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
for key in ("train_flow_only", "train_flow_lora"):
    if key not in fields:
        raise RuntimeError(f"freeze_policy missing {key}")

source = getsource(TrainerUtils.load_finetune_init_weights)
for marker in (
    'key.startswith("policy_backend.flow.")',
    'key.startswith("policy_action_head.")',
    "if (not load_pretrained_policy_flow) and _is_policy_flow_key(key)",
):
    if marker not in source:
        raise RuntimeError(
            "Current loader cannot prove a fresh Full Flow because the "
            f"required alias-aware skip marker is missing: {marker}"
        )
print("[OK] strict Flow-only mode and alias-aware fresh-Flow loading detected.")
PY

RUN_ID="t9_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_full_flow_only_scratch"
PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/full_flow_t9_scratch_${PIPELINE_TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " LaWAM Goal: Strict Frozen Upstream + Scratch Full Flow, T9"
echo "=========================================================="
echo "Base          : ${BASE_RUN}"
echo "Task          : T9"
echo "Train GPUs    : ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "batch/GPU     : ${PER_DEVICE_BATCH_SIZE}"
echo "grad accum    : ${GRADIENT_ACCUMULATION_STEPS}"
echo "global batch  : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_GPUS))"
echo "steps         : ${MAX_TRAIN_STEPS}"
echo "warmup        : ${NUM_WARMUP_STEPS}"
echo "action LR     : ${ACTION_LR}"
echo "seed          : ${SEED}"
echo "repeat FM     : ${REPEATED_DIFFUSION_STEPS}"
echo "Trainable     : COMPLETE policy_backend.flow.* ONLY"
echo "Flow capacity : hidden=1024, layers=16, heads=16, FFN=4096"
echo "Flow init     : RANDOM; both Base Flow aliases are skipped"
echo "Upstream init : loaded exactly from Formal Base"
echo "Evaluation    : T9, ${NUM_TRIALS} trials"
echo "Run ID        : ${RUN_ID}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_TRAIN_GPUS}"

bash train_lawam.sh \
    "--run_root_dir=${RUN_ROOT}" \
    "--run_id=${RUN_ID}" \
    "--datasets.vla_data.cl_suite=libero_goal" \
    "--datasets.vla_data.cl_task_ids=[9]" \
    "--datasets.vla_data.use_task_filtered_statistics=false" \
    "--seed=${SEED}" \
    "--trainer.use_pretrained_dataset_statistics=true" \
    "--trainer.pretrained_checkpoint=${BASE_CKPT}" \
    "--trainer.load_pretrained_policy_flow=false" \
    "--trainer.strict_finetune_init=true" \
    "--framework.action_model.flow_cfg.expert_variant=full" \
    "--framework.action_model.flow_cfg.hidden_dim=1024" \
    "--framework.action_model.flow_cfg.num_layers=16" \
    "--framework.action_model.flow_cfg.attention_heads=16" \
    "--framework.action_model.flow_cfg.ffn_dim=4096" \
    "--framework.action_model.repeated_diffusion_steps=${REPEATED_DIFFUSION_STEPS}" \
    "--framework.action_model.enable_loss_distill=false" \
    "--framework.action_model.perceptual_weight=0.0" \
    "--framework.action_model.lam_encoder_distill_weight=0.0" \
    "--trainer.freeze.train_flow_only=true" \
    "--trainer.freeze.train_flow_lora=false" \
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

unset CUDA_VISIBLE_DEVICES || true
unset NUM_PROCESSES || true

RUN=$(find_run "${RUN_ROOT}" "*+${RUN_ID}")
verify_run "Scratch Full-Flow T9" "${RUN}"

# Require positive evidence in the training log that both Flow aliases were
# intentionally skipped. This catches an old loader or an accidentally flipped
# command-line flag before the result is interpreted as a scratch control.
python - "${MASTER_LOG}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
required = (
    "Finetune init skipped both Flow aliases",
    "load_pretrained_policy_flow=False",
    "intentionally_skipped_flow_keys=",
)
missing = [marker for marker in required if marker not in text]
if missing:
    raise RuntimeError(
        f"Fresh Full-Flow initialization audit failed for {path}: "
        f"missing log markers={missing}"
    )
matches = re.findall(r"intentionally_skipped_flow_keys=(\d+)", text)
if not matches or max(map(int, matches)) <= 0:
    raise RuntimeError("No positive intentionally_skipped_flow_keys count found")
print(
    "[OK] initialization audit: both Flow aliases skipped; "
    f"max_intentionally_skipped_flow_keys={max(map(int, matches))}."
)
PY

python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json
import sys

base_path, current_path = sys.argv[1:3]
with open(base_path, "r", encoding="utf-8") as f:
    base = json.load(f)
with open(current_path, "r", encoding="utf-8") as f:
    current = json.load(f)
for tag in base:
    for section in ("action", "state"):
        if base[tag][section] != current[tag][section]:
            raise RuntimeError(f"Normalization statistics changed: {tag}/{section}")
print("[OK] action/state normalization is identical to Base.")
PY

python - "${RUN}/config.yaml" "${BASE_CKPT}" "${ACTION_LR}" "${SEED}" "${REPEATED_DIFFUSION_STEPS}" <<'PY'
import math
import sys

from omegaconf import OmegaConf

path, base, lr_string, seed_string, repeat_string = sys.argv[1:6]
lr = float(lr_string)
cfg = OmegaConf.load(path)
freeze = cfg.trainer.freeze

if not bool(freeze.get("train_flow_only", False)):
    raise RuntimeError("train_flow_only is not True")
if bool(freeze.get("train_flow_lora", False)):
    raise RuntimeError("train_flow_lora must be False")
if bool(freeze.get("unfreeze_lam_decoder", False)):
    raise RuntimeError("unfreeze_lam_decoder must be False")
if str(cfg.trainer.pretrained_checkpoint) != str(base):
    raise RuntimeError("Frozen upstream is not initialized from Formal Base")
if list(cfg.datasets.vla_data.cl_task_ids) != [9]:
    raise RuntimeError("Wrong task filter")
if bool(cfg.trainer.load_pretrained_policy_flow):
    raise RuntimeError("load_pretrained_policy_flow must be False for Full-scratch")
if not bool(cfg.trainer.strict_finetune_init):
    raise RuntimeError("strict_finetune_init must be True")

flow = cfg.framework.action_model.flow_cfg
expected = ("full", 1024, 16, 16, 4096)
actual = (
    str(flow.expert_variant),
    int(flow.hidden_dim),
    int(flow.num_layers),
    int(flow.attention_heads),
    int(flow.ffn_dim),
)
if actual != expected:
    raise RuntimeError(
        f"Unexpected Full Flow architecture: expected={expected}, actual={actual}"
    )
if int(cfg.seed) != int(seed_string):
    raise RuntimeError("Unexpected seed")
if int(cfg.framework.action_model.repeated_diffusion_steps) != int(repeat_string):
    raise RuntimeError("Unexpected repeated_diffusion_steps")
if bool(cfg.framework.action_model.enable_loss_distill):
    raise RuntimeError("enable_loss_distill must be False")
if float(cfg.framework.action_model.perceptual_weight) != 0.0:
    raise RuntimeError("perceptual_weight must be zero")
if float(cfg.framework.action_model.lam_encoder_distill_weight) != 0.0:
    raise RuntimeError("lam_encoder_distill_weight must be zero")
if not math.isclose(
    float(cfg.trainer.learning_rate.action_model.lr),
    lr,
    rel_tol=0,
    abs_tol=1e-12,
):
    raise RuntimeError("Unexpected action_model LR")
print("[OK] Strict Scratch Full-Flow T9 training config verified.")
PY

python - "${BASE_CKPT}" "${RUN}/final_model/pytorch_model.pt" <<'PY'
import sys

import torch

base_path, current_path = sys.argv[1:3]


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


def is_flow(key):
    return key.startswith("policy_backend.flow.") or key.startswith(
        "policy_action_head."
    )


base = load(base_path)
current = load(current_path)
base_keys = {key for key, value in base.items() if torch.is_tensor(value)}
current_keys = {key for key, value in current.items() if torch.is_tensor(value)}
if base_keys != current_keys:
    raise RuntimeError(
        f"checkpoint key mismatch missing={sorted(base_keys-current_keys)[:20]} "
        f"extra={sorted(current_keys-base_keys)[:20]}"
    )

changed_upstream = []
canonical_changed = []
alias_changed = []
for key in sorted(base_keys):
    base_value = base[key]
    current_value = current[key]
    changed = (
        tuple(base_value.shape) != tuple(current_value.shape)
        or base_value.dtype != current_value.dtype
        or not torch.equal(base_value, current_value)
    )
    if not changed:
        continue
    if key.startswith("policy_backend.flow."):
        canonical_changed.append(key)
    elif key.startswith("policy_action_head."):
        alias_changed.append(key)
    else:
        changed_upstream.append(key)

if changed_upstream:
    raise RuntimeError(
        "FULL-SCRATCH ISOLATION FAILED: frozen upstream changed "
        f"count={len(changed_upstream)} examples={changed_upstream[:30]}"
    )
if not canonical_changed:
    raise RuntimeError("No canonical Flow tensor changed; scratch training likely failed")

base_canonical = {
    key[len("policy_backend.flow.") :]: value
    for key, value in base.items()
    if key.startswith("policy_backend.flow.") and torch.is_tensor(value)
}
current_canonical = {
    key[len("policy_backend.flow.") :]: value
    for key, value in current.items()
    if key.startswith("policy_backend.flow.") and torch.is_tensor(value)
}
current_alias = {
    key[len("policy_action_head.") :]: value
    for key, value in current.items()
    if key.startswith("policy_action_head.") and torch.is_tensor(value)
}
if set(base_canonical) != set(current_canonical):
    raise RuntimeError("Scratch Full Flow does not have the Base Full-Flow key set")
shape_mismatch = [
    suffix
    for suffix in base_canonical
    if tuple(base_canonical[suffix].shape) != tuple(current_canonical[suffix].shape)
]
if shape_mismatch:
    raise RuntimeError(
        f"Scratch Full Flow architecture differs from Base: {shape_mismatch[:20]}"
    )
if set(current_alias) != set(current_canonical):
    raise RuntimeError("Final checkpoint Flow alias key set is inconsistent")
alias_mismatch = [
    suffix
    for suffix in current_canonical
    if not torch.equal(current_canonical[suffix], current_alias[suffix])
]
if alias_mismatch:
    raise RuntimeError(
        f"Final checkpoint Flow aliases disagree: {alias_mismatch[:20]}"
    )

upstream_checked = sum(1 for key in base_keys if not is_flow(key))
canonical_total = len(current_canonical)
alias_total = len(current_alias)
canonical_numel = sum(value.numel() for value in current_canonical.values())
print(
    "[full-scratch-check] T9: "
    f"upstream_checked={upstream_checked}, upstream_changed=0, "
    f"canonical_flow_checked={canonical_total}, "
    f"canonical_flow_changed={len(canonical_changed)}, "
    f"flow_alias_checked={alias_total}, flow_alias_changed={len(alias_changed)}, "
    f"canonical_flow_numel={canonical_numel:,}, exact_upstream=True, "
    "same_full_architecture=True"
)
PY

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

EVAL_MASTER="${OUTPUT_ROOT}/t9_${PIPELINE_TIMESTAMP}"
ALIAS="full_flow_t9_scratch"
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
[ -n "${EVAL_DIR}" ] || { echo "[ERROR] T9 evaluation output not found"; exit 1; }
SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
python "${SUMMARY_SCRIPT}" \
    --run-dir "${SUITE_DIR}" \
    --task-ids 9 \
    --expected-trials "${NUM_TRIALS}"

echo "${RUN}/final_model/pytorch_model.pt" > "${EXPERIMENT_ROOT}/LATEST_T9_FULL_FLOW_SCRATCH.txt"
echo "${SUITE_DIR}/per_task_summary.csv" > "${EXPERIMENT_ROOT}/LATEST_T9_FULL_FLOW_SCRATCH_SUMMARY.txt"

echo "=========================================================="
echo " Scratch Full-Flow T9 control complete"
echo "=========================================================="
echo "Checkpoint: ${RUN}/final_model/pytorch_model.pt"
echo "Pointer   : ${EXPERIMENT_ROOT}/LATEST_T9_FULL_FLOW_SCRATCH.txt"
echo "Evaluation: ${SUITE_DIR}/per_task_summary.csv"
echo "Log       : ${MASTER_LOG}"
echo "=========================================================="
