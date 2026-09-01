#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LaWAM Goal T9 Flow-LoRA Level-3 variant
#
# Level-2 base targets ALWAYS ON:
#   - Transformer Q/K/V/O + FFN
#   - enc_vlm
#   - DiT proj_out_1/proj_out_2
#
# Optional Level-3:
#   - action_encoder.W1/W2
#   - action_decoder.layer1/layer2
#
# Every original Base tensor remains frozen.
# ONLY lora_A / lora_B are trainable.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
MERGE_SCRIPT="${ROOT}/scripts/merge_flow_lora_level3_checkpoint.py"

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LR="${LORA_LR:-0.0001}"

LORA_TARGET_ACTION_ENCODER="${LORA_TARGET_ACTION_ENCODER:-false}"
LORA_TARGET_ACTION_DECODER="${LORA_TARGET_ACTION_DECODER:-false}"
TARGET_TAG="${TARGET_TAG:-level3_custom}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"

IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"

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

sanitize_tag() {
    echo "$1" | sed -e 's/\./p/g' -e 's/-/m/g' -e 's/+//g'
}

RANK_TAG=$(sanitize_tag "${LORA_RANK}")
ALPHA_TAG=$(sanitize_tag "${LORA_ALPHA}")
LR_TAG=$(sanitize_tag "${LORA_LR}")

EXPERIMENT_TAG="r${RANK_TAG}_a${ALPHA_TAG}_lr${LR_TAG}_${TARGET_TAG}"

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_lora_level3_t9/${EXPERIMENT_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_lora_level3_t9/${EXPERIMENT_TAG}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

find_run() {
    local root="$1"
    local pattern="$2"
    find "${root}" -maxdepth 1 -type d -name "${pattern}" \
        | sort | tail -n 1
}

verify_run() {
    local label="$1"
    local run="$2"

    [ -n "${run}" ] || {
        echo "[ERROR] ${label}: run not found"
        exit 1
    }

    for f in \
        "${run}/config.yaml" \
        "${run}/dataset_statistics.json" \
        "${run}/final_model/pytorch_model.pt"
    do
        [ -f "${f}" ] || {
            echo "[ERROR] ${label}: missing ${f}"
            exit 1
        }
    done

    echo "[OK] ${label}: ${run}"
}

# -------------------------------------------------------------------------
# Code preflight.
# -------------------------------------------------------------------------
python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)

fields = set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required = {
    "train_flow_lora",
    "flow_lora_target_enc_vlm",
    "flow_lora_target_output",
    "flow_lora_target_action_encoder",
    "flow_lora_target_action_decoder",
}

missing = sorted(required - fields)
if missing:
    raise RuntimeError(
        f"Level-3 Flow-LoRA support missing: {missing}"
    )

print("[OK] Flow-LoRA Level-3 support detected.")
PY

if [ -n "${BASE_RUN:-}" ]; then
    :
else
    BASE_RUN=$(find_run \
        "${BASE_ROOT}" \
        '*+base_t0_5_10k_4gpu_bs32_ga2')
fi

verify_run "Formal Goal Base" "${BASE_RUN}"

BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

RUN_ID="t9_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_${EXPERIMENT_TAG}_from_base"

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/t9_${PIPELINE_TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " LaWAM Goal T9 Flow-LoRA Level-3"
echo "=========================================================="
echo "Base           : ${BASE_RUN}"
echo "rank/alpha     : ${LORA_RANK}/${LORA_ALPHA}"
echo "LoRA LR        : ${LORA_LR}"
echo "Level-2 core   : Transformer + enc_vlm + output = ON"
echo "Action encoder : ${LORA_TARGET_ACTION_ENCODER}"
echo "Action decoder : ${LORA_TARGET_ACTION_DECODER}"
echo "Category LoRA  : shared task residual"
echo "Target tag     : ${TARGET_TAG}"
echo "Train GPUs     : ${TRAIN_GPUS}"
echo "Global batch   : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_GPUS))"
echo "Steps          : ${MAX_TRAIN_STEPS}"
echo "Trials         : ${NUM_TRIALS}"
echo "=========================================================="

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NUM_PROCESSES="${NUM_TRAIN_GPUS}"

bash train_lawam.sh \
    "--run_root_dir=${RUN_ROOT}" \
    "--run_id=${RUN_ID}" \
    "--datasets.vla_data.cl_suite=libero_goal" \
    "--datasets.vla_data.cl_task_ids=[9]" \
    "--datasets.vla_data.use_task_filtered_statistics=false" \
    "--trainer.use_pretrained_dataset_statistics=true" \
    "--trainer.pretrained_checkpoint=${BASE_CKPT}" \
    "--trainer.load_pretrained_policy_flow=true" \
    "--trainer.freeze.train_flow_only=false" \
    "--trainer.freeze.train_flow_lora=true" \
    "--trainer.freeze.flow_lora_rank=${LORA_RANK}" \
    "--trainer.freeze.flow_lora_alpha=${LORA_ALPHA}" \
    "--trainer.freeze.flow_lora_dropout=${LORA_DROPOUT}" \
    "--trainer.freeze.flow_lora_target_attention=true" \
    "--trainer.freeze.flow_lora_target_ffn=true" \
    "--trainer.freeze.flow_lora_target_enc_vlm=true" \
    "--trainer.freeze.flow_lora_target_output=true" \
    "--trainer.freeze.flow_lora_target_action_encoder=${LORA_TARGET_ACTION_ENCODER}" \
    "--trainer.freeze.flow_lora_target_action_decoder=${LORA_TARGET_ACTION_DECODER}" \
    "--trainer.freeze.unfreeze_lam_decoder=false" \
    "--trainer.learning_rate.vlm.lr=${LORA_LR}" \
    "--trainer.learning_rate.action_model.lr=${LORA_LR}" \
    "--trainer.learning_rate.world_model.lr=${LORA_LR}" \
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
verify_run "T9 ${TARGET_TAG}" "${RUN}"

UNMERGED="${RUN}/final_model/pytorch_model.pt"
MERGED="${RUN}/final_model/pytorch_model_merged.pt"
ADAPTER="${RUN}/final_model/flow_lora_adapter.pt"

# -------------------------------------------------------------------------
# Normalization audit.
# -------------------------------------------------------------------------
python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    base = json.load(f)
with open(sys.argv[2], "r", encoding="utf-8") as f:
    cur = json.load(f)

for tag in base:
    for sec in ("action", "state"):
        if base[tag][sec] != cur[tag][sec]:
            raise RuntimeError(
                f"Normalization statistics changed: {tag}/{sec}"
            )

print("[OK] action/state normalization identical to Base.")
PY

# -------------------------------------------------------------------------
# Config audit.
# -------------------------------------------------------------------------
python - \
    "${RUN}/config.yaml" \
    "${BASE_CKPT}" \
    "${LORA_RANK}" \
    "${LORA_ALPHA}" \
    "${LORA_LR}" \
    "${LORA_TARGET_ACTION_ENCODER}" \
    "${LORA_TARGET_ACTION_DECODER}" <<'PY'
import math
import sys

from omegaconf import OmegaConf

path, base, rank_s, alpha_s, lr_s, enc_s, dec_s = sys.argv[1:8]

def as_bool(x):
    return str(x).lower() == "true"

cfg = OmegaConf.load(path)
fr = cfg.trainer.freeze

expected = {
    "train_flow_only": False,
    "train_flow_lora": True,
    "flow_lora_rank": int(rank_s),
    "flow_lora_alpha": float(alpha_s),
    "flow_lora_target_attention": True,
    "flow_lora_target_ffn": True,
    "flow_lora_target_enc_vlm": True,
    "flow_lora_target_output": True,
    "flow_lora_target_action_encoder": as_bool(enc_s),
    "flow_lora_target_action_decoder": as_bool(dec_s),
    "unfreeze_lam_decoder": False,
}

for key, expected_value in expected.items():
    actual = fr.get(key, None)

    if isinstance(expected_value, float):
        if not math.isclose(
            float(actual),
            expected_value,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{key}: {actual} != {expected_value}"
            )
    elif actual != expected_value:
        raise RuntimeError(
            f"{key}: {actual} != {expected_value}"
        )

if str(cfg.trainer.pretrained_checkpoint) != str(base):
    raise RuntimeError(
        "Training did not initialize from formal Base."
    )

if list(cfg.datasets.vla_data.cl_task_ids) != [9]:
    raise RuntimeError("Wrong task filter; expected [9].")

if not math.isclose(
    float(cfg.trainer.learning_rate.action_model.lr),
    float(lr_s),
    rel_tol=0,
    abs_tol=1e-12,
):
    raise RuntimeError("Unexpected action-model LR.")

print("[OK] Level-3 Flow-LoRA config verified.")
PY

# -------------------------------------------------------------------------
# HARD unmerged checkpoint audit.
# Every Base tensor must remain bitwise identical.
# -------------------------------------------------------------------------
python - \
    "${BASE_CKPT}" \
    "${UNMERGED}" \
    "${LORA_RANK}" \
    "${LORA_TARGET_ACTION_ENCODER}" \
    "${LORA_TARGET_ACTION_DECODER}" <<'PY'
import re
import sys
import torch

base_path, cur_path, rank_s, enc_s, dec_s = sys.argv[1:6]
rank = int(rank_s)
enc_enabled = str(enc_s).lower() == "true"
dec_enabled = str(dec_s).lower() == "true"

def load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(path, map_location="cpu")

base = load(base_path)
cur = load(cur_path)

base_keys = {
    k for k, v in base.items()
    if torch.is_tensor(v)
}
cur_keys = {
    k for k, v in cur.items()
    if torch.is_tensor(v)
}

missing = sorted(base_keys - cur_keys)
if missing:
    raise RuntimeError(
        f"Base tensor keys missing: {missing[:30]}"
    )

changed_base = []
for key in sorted(base_keys):
    a = base[key]
    b = cur[key]

    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    ):
        changed_base.append(key)

if changed_base:
    raise RuntimeError(
        "LEVEL-3 FREEZE FAILED: original Base tensors changed. "
        f"count={len(changed_base)}, "
        f"examples={changed_base[:30]}"
    )

extra = sorted(cur_keys - base_keys)

bad_extra = [
    k for k in extra
    if not (
        k.endswith(".lora_A")
        or k.endswith(".lora_B")
    )
]
if bad_extra:
    raise RuntimeError(
        f"Unexpected non-LoRA extra tensors: {bad_extra[:30]}"
    )

canonical_a = sorted(
    k for k in extra
    if (
        k.startswith("policy_backend.flow.")
        and k.endswith(".lora_A")
    )
)
canonical_b = sorted(
    k for k in extra
    if (
        k.startswith("policy_backend.flow.")
        and k.endswith(".lora_B")
    )
)

alias_a = sorted(
    k for k in extra
    if (
        k.startswith("policy_action_head.")
        and k.endswith(".lora_A")
    )
)
alias_b = sorted(
    k for k in extra
    if (
        k.startswith("policy_action_head.")
        and k.endswith(".lora_B")
    )
)

if len(canonical_a) != len(canonical_b):
    raise RuntimeError("Canonical LoRA A/B count mismatch.")

if (
    len(alias_a) != len(canonical_a)
    or len(alias_b) != len(canonical_b)
):
    raise RuntimeError(
        "Canonical/alias LoRA count mismatch."
    )

def group_from_a_key(key):
    suffix = key[len("policy_backend.flow."):]
    prefix = suffix[:-len(".lora_A")]

    if re.match(
        r"^DiT\.transformer_blocks\.\d+\.attn1\.",
        prefix,
    ):
        return "attention"

    if re.match(
        r"^DiT\.transformer_blocks\.\d+\.ff\.",
        prefix,
    ):
        return "ffn"

    if prefix.startswith("enc_vlm."):
        return "enc_vlm"

    if prefix in (
        "DiT.proj_out_1",
        "DiT.proj_out_2",
    ):
        return "output"

    if prefix in (
        "action_encoder.W1",
        "action_encoder.W2",
    ):
        return "action_encoder"

    if prefix in (
        "action_decoder.layer1",
        "action_decoder.layer2",
    ):
        return "action_decoder"

    return "unexpected"

groups = {
    "attention": 0,
    "ffn": 0,
    "enc_vlm": 0,
    "output": 0,
    "action_encoder": 0,
    "action_decoder": 0,
}

unexpected = []

for key in canonical_a:
    group = group_from_a_key(key)

    if group == "unexpected":
        unexpected.append(key)
    else:
        groups[group] += 1

if unexpected:
    raise RuntimeError(
        f"LoRA injected outside Level-3 targets: {unexpected[:30]}"
    )

expected_counts = {
    "attention": 64,
    "ffn": 32,
    "enc_vlm": 1,
    "output": 2,
    "action_encoder": 2 if enc_enabled else 0,
    "action_decoder": 2 if dec_enabled else 0,
}

for group, expected in expected_counts.items():
    if groups[group] != expected:
        raise RuntimeError(
            f"{group}: expected {expected}, got {groups[group]}"
        )

# Rank-shape audit:
# standard: A [r,in]
# category: A [in,r]
for key in canonical_a:
    prefix = key[:-len(".lora_A")]
    group = group_from_a_key(key)

    if group in ("action_encoder", "action_decoder"):
        if int(cur[key].shape[1]) != rank:
            raise RuntimeError(
                f"Category LoRA rank mismatch: {key} {tuple(cur[key].shape)}"
            )
    else:
        if int(cur[key].shape[0]) != rank:
            raise RuntimeError(
                f"Standard LoRA rank mismatch: {key} {tuple(cur[key].shape)}"
            )

canonical_lora_params = sum(
    cur[k].numel()
    for k in canonical_a + canonical_b
)

b_abs_sum = sum(
    float(cur[k].float().abs().sum().item())
    for k in canonical_b
)

if b_abs_sum <= 0.0:
    raise RuntimeError(
        "All canonical LoRA-B tensors remain zero; adapter did not train."
    )

print(
    "[level3-lora-check] "
    f"base_checked={len(base_keys)}, "
    "base_changed=0, "
    f"canonical_modules={len(canonical_a)}, "
    f"canonical_lora_params={canonical_lora_params:,}, "
    f"group_modules={groups}, "
    f"B_abs_sum={b_abs_sum:.6g}, "
    "exact_base=True"
)
PY

# -------------------------------------------------------------------------
# Merge adapter into standard LaWAM checkpoint.
# -------------------------------------------------------------------------
python "${MERGE_SCRIPT}" \
    "${UNMERGED}" \
    --output "${MERGED}" \
    --adapter-output "${ADAPTER}" \
    --alpha "${LORA_ALPHA}"

# -------------------------------------------------------------------------
# HARD merged audit.
# Only actual LoRA base weight/W tensors may differ.
# -------------------------------------------------------------------------
python - \
    "${BASE_CKPT}" \
    "${UNMERGED}" \
    "${MERGED}" <<'PY'
import sys
import torch

base_path, unmerged_path, merged_path = sys.argv[1:4]

def load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            return torch.load(path, map_location="cpu")

base = load(base_path)
unmerged = load(unmerged_path)
merged = load(merged_path)

base_keys = {
    k for k, v in base.items()
    if torch.is_tensor(v)
}
merged_keys = {
    k for k, v in merged.items()
    if torch.is_tensor(v)
}

if base_keys != merged_keys:
    raise RuntimeError(
        "Merged checkpoint key mismatch. "
        f"missing={sorted(base_keys-merged_keys)[:20]}, "
        f"extra={sorted(merged_keys-base_keys)[:20]}"
    )

allowed = set()

for key, value in unmerged.items():
    if not (
        torch.is_tensor(value)
        and key.endswith(".lora_A")
    ):
        continue

    prefix = key[:-len(".lora_A")]

    if prefix + ".weight" in unmerged:
        allowed.add(prefix + ".weight")
    elif prefix + ".W" in unmerged:
        allowed.add(prefix + ".W")
    else:
        raise RuntimeError(
            f"Cannot resolve Base target for {prefix}"
        )

changed = []
bad = []

for key in sorted(base_keys):
    a = base[key]
    b = merged[key]

    if (
        tuple(a.shape) != tuple(b.shape)
        or a.dtype != b.dtype
        or not torch.equal(a, b)
    ):
        changed.append(key)

        if key not in allowed:
            bad.append(key)

if bad:
    raise RuntimeError(
        "Merged checkpoint changed non-target Base tensors: "
        f"{bad[:30]}"
    )

if not changed:
    raise RuntimeError(
        "Merged checkpoint has no changed LoRA targets."
    )

print(
    "[level3-lora-merged-check] "
    f"changed_target_tensors={len(changed)}, "
    "changed_non_target=0, "
    "standard_checkpoint_keys=True"
)
PY

# -------------------------------------------------------------------------
# T9 rollout.
# -------------------------------------------------------------------------
export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

EVAL_MASTER="${OUTPUT_ROOT}/t9_${PIPELINE_TIMESTAMP}"
ALIAS="level3_${EXPERIMENT_TAG}_T9"

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

EVAL_DIR=$(find \
    "${EVAL_MASTER}/${ALIAS}" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    | sort \
    | tail -n 1)

[ -n "${EVAL_DIR}" ] || {
    echo "[ERROR] T9 evaluation output not found"
    exit 1
}

SUITE_DIR="${EVAL_DIR}/suites/libero_goal"

python "${SUMMARY_SCRIPT}" \
    --run-dir "${SUITE_DIR}" \
    --task-ids 9 \
    --expected-trials "${NUM_TRIALS}"

RESULT="${SUITE_DIR}/per_task_summary.csv"

[ -f "${RESULT}" ] || {
    echo "[ERROR] Missing ${RESULT}"
    exit 1
}

echo "${RESULT}" > "${EXPERIMENT_ROOT}/LATEST_RESULT.txt"
echo "${RUN}" > "${EXPERIMENT_ROOT}/LATEST_RUN.txt"

echo
echo "=========================================================="
echo " Level-3 T9 Flow-LoRA complete"
echo "=========================================================="
echo "Setting   : ${TARGET_TAG}"
echo "Run       : ${RUN}"
echo "Adapter   : ${ADAPTER}"
echo "Merged    : ${MERGED}"
echo "Result    : ${RESULT}"
echo "Log       : ${MASTER_LOG}"
echo "=========================================================="
