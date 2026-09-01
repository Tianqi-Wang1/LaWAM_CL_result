#!/usr/bin/env bash

set -euo pipefail


# ==========================================================
# LaWAM LIBERO-Object Strict Freeze-VLM Continual Learning
#
# Experimental definition:
#
#   Reuse EXACT SeqFT Base checkpoint (tasks 0-5).
#   Only CL1-CL4 change the training policy.
#
#   Frozen during CL:
#     - Qwen vision backbone
#     - Qwen vision merger
#     - Qwen LLM backbone
#     - Qwen embeddings
#
#   Still trainable:
#     - act_query
#     - flow_action_query
#     - VLMToLAM / QFormer
#     - LaWM decoder
#     - Flow action head
#
# Sequential protocol:
#   Base : object 0-5 (reuse existing 10K SeqFT Base)
#   CL1  : object 6, 2K
#   CL2  : object 7, 2K
#   CL3  : object 8, 2K
#   CL4  : object 9, 2K
#
# The script performs:
#   1) sequential CL1 -> CL4 training
#   2) Base-statistics consistency checks
#   3) freeze-config checks
#   4) checkpoint-level VLM invariance check
#   5) full Base/CL1/CL2/CL3/CL4 LIBERO evaluation
#   6) CL performance-matrix aggregation
#
# Default launch:
#   bash scripts/run_libero_object_freeze_vlm_full_v1.sh
#
# Resume training from CL3:
#   START_FROM=cl CL_START_STAGE=3 bash ...
#
# Eval only:
#   START_FROM=eval bash ...
# ==========================================================


# ==========================================================
# 0. Environment
# ==========================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh

conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"


# ==========================================================
# 1. Paths
# ==========================================================

# Existing vanilla SeqFT Base lives here and is never modified.
BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"

# Freeze-VLM CL1-CL4 are isolated from vanilla SeqFT checkpoints.
RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/freeze_vlm"

# Evaluation outputs are isolated as well.
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_object/freeze_vlm_full"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"

mkdir -p "${RUN_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

if [ ! -f "${SUMMARY_SCRIPT}" ]; then
    echo "[ERROR] Missing summary script:"
    echo "        ${SUMMARY_SCRIPT}"
    exit 1
fi


# ==========================================================
# 2. Resource configuration
# ==========================================================

# Training: exactly four physical GPUs.
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"

# Evaluation: one policy GPU + one simulator/evaluator GPU.
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"

EVAL_WORKERS="${EVAL_WORKERS:-16}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"

if [ "${#TRAIN_GPU_ARRAY[@]}" -ne 4 ]; then
    echo "[ERROR] Exactly four training GPUs are required."
    echo "        TRAIN_GPUS=${TRAIN_GPUS}"
    exit 1
fi


# ==========================================================
# 3. Training hyperparameters
#    Keep identical to the formal SeqFT CL protocol.
# ==========================================================

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"

NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"

LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"


# ==========================================================
# 4. Resume control
#
# START_FROM:
#   cl   -> run CL training, then full evaluation
#   eval -> skip training and evaluate existing freeze-VLM chain
#
# CL_START_STAGE:
#   1,2,3,4
# ==========================================================

START_FROM="${START_FROM:-cl}"
CL_START_STAGE="${CL_START_STAGE:-1}"

case "${START_FROM}" in
    cl|eval)
        ;;
    *)
        echo "[ERROR] START_FROM must be cl or eval."
        exit 1
        ;;
esac

if ! [[ "${CL_START_STAGE}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] CL_START_STAGE must be 1, 2, 3, or 4."
    exit 1
fi


# ==========================================================
# 5. Runtime environment
# ==========================================================

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


# ==========================================================
# 6. Helpers: resolve and verify runs
# ==========================================================

find_run() {
    local root="$1"
    local pattern="$2"

    find "${root}" \
        -maxdepth 1 \
        -type d \
        -name "${pattern}" \
        | sort \
        | tail -n 1
}


verify_run() {
    local name="$1"
    local run="$2"

    if [ -z "${run}" ]; then
        echo "[ERROR] ${name} run was not found."
        exit 1
    fi

    if [ ! -f "${run}/final_model/pytorch_model.pt" ]; then
        echo "[ERROR] ${name} checkpoint missing:"
        echo "        ${run}/final_model/pytorch_model.pt"
        exit 1
    fi

    if [ ! -f "${run}/dataset_statistics.json" ]; then
        echo "[ERROR] ${name} dataset statistics missing:"
        echo "        ${run}/dataset_statistics.json"
        exit 1
    fi

    echo "[OK] ${name}"
    echo "     run   : ${run}"
    echo "     ckpt  : ${run}/final_model/pytorch_model.pt"
    echo "     stats : ${run}/dataset_statistics.json"
}


# ==========================================================
# 7. Locate the EXISTING vanilla SeqFT Base
# ==========================================================

if [ -n "${BASE_RUN:-}" ]; then
    echo "[INFO] Using explicitly provided BASE_RUN:"
    echo "       ${BASE_RUN}"
else
    BASE_RUN=$(find_run \
        "${BASE_ROOT}" \
        '*+base_t0_5_10k_4gpu_bs32_ga2'
    )
fi

verify_run "Existing Object SeqFT Base" "${BASE_RUN}"

BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"


# ==========================================================
# 8. Verify Base normalization remains fixed
# ==========================================================

verify_statistics() {
    local reference_stats="$1"
    local current_stats="$2"

    python - "${reference_stats}" "${current_stats}" <<'PY'
import json
import sys

reference_path, current_path = sys.argv[1:3]

with open(reference_path, "r", encoding="utf-8") as f:
    reference = json.load(f)

with open(current_path, "r", encoding="utf-8") as f:
    current = json.load(f)

for tag in reference:
    if tag not in current:
        raise RuntimeError(f"Missing embodiment tag in current statistics: {tag}")

    for section in ("action", "state"):
        if section not in reference[tag]:
            raise RuntimeError(f"Missing reference statistics section: {tag}/{section}")
        if section not in current[tag]:
            raise RuntimeError(f"Missing current statistics section: {tag}/{section}")
        if reference[tag][section] != current[tag][section]:
            raise RuntimeError(f"Normalization statistics changed: {tag}/{section}")

print("[OK] action/state normalization is identical to Base statistics.")
PY
}


# ==========================================================
# 9. Verify the saved training config really used strict
#    Freeze-VLM.
# ==========================================================

verify_freeze_config() {
    local config_path="$1"

    if [ ! -f "${config_path}" ]; then
        echo "[ERROR] Missing config.yaml:"
        echo "        ${config_path}"
        exit 1
    fi

    python - "${config_path}" <<'PY'
import sys
from omegaconf import OmegaConf

path = sys.argv[1]
cfg = OmegaConf.load(path)
freeze = cfg.trainer.freeze

expected = {
    "freeze_vision_backbone": True,
    "freeze_llm_backbone": True,
    "freeze_last_llm_layer": True,
    "freeze_embedding": True,
    "unfreeze_vision_merger": False,
    "keep_llm_first_n_layers": 16,
    "unfreeze_llm_last_n_layers": -1,
    "unfreeze_lam_decoder": True,
}

bad = []
for key, expected_value in expected.items():
    actual = freeze.get(key, None)
    if actual != expected_value:
        bad.append((key, actual, expected_value))

if bad:
    lines = "\n".join(
        f"  {k}: actual={a!r}, expected={e!r}"
        for k, a, e in bad
    )
    raise RuntimeError(
        "Strict Freeze-VLM config mismatch:\n" + lines
    )

print("[OK] strict Freeze-VLM config verified.")
for key, expected_value in expected.items():
    print(f"     {key}: {expected_value}")
PY
}


# ==========================================================
# 10. Checkpoint-level parameter change diagnostic
#
# HARD ASSERTION:
#   policy_backend.vlm.* must remain bitwise identical.
#
# REPORT ONLY:
#   act_query
#   flow_action_query
#   vlm_to_lam
#   LaWM decoder
#   Flow head
#
# This is intentionally done after each CL stage so a broken
# freeze policy cannot silently contaminate the whole chain.
# ==========================================================

verify_checkpoint_freeze() {
    local previous_ckpt="$1"
    local current_ckpt="$2"
    local stage_name="$3"

    python - "${previous_ckpt}" "${current_ckpt}" "${stage_name}" <<'PY'
import gc
import sys
import torch

prev_path, cur_path, stage = sys.argv[1:4]


def load_state(path):
    kwargs = dict(map_location="cpu")
    try:
        obj = torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            obj = torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            obj = torch.load(path, **kwargs)

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            nested = obj.get(key, None)
            if isinstance(nested, dict) and nested:
                # Only unwrap when the nested dict itself looks like a state dict.
                if any(torch.is_tensor(v) for v in nested.values()):
                    obj = nested
                    break
    if not isinstance(obj, dict):
        raise RuntimeError(f"Checkpoint is not a state dict: {path}")
    return obj


prev = load_state(prev_path)
cur = load_state(cur_path)


def keys_for(group):
    if group.endswith("."):
        return {
            k for k, v in prev.items()
            if k.startswith(group) and torch.is_tensor(v)
        }, {
            k for k, v in cur.items()
            if k.startswith(group) and torch.is_tensor(v)
        }

    return {
        k for k, v in prev.items()
        if (k == group or k.startswith(group + ".")) and torch.is_tensor(v)
    }, {
        k for k, v in cur.items()
        if (k == group or k.startswith(group + ".")) and torch.is_tensor(v)
    }


def compare(group):
    pkeys, ckeys = keys_for(group)
    missing = sorted(pkeys - ckeys)
    extra = sorted(ckeys - pkeys)
    common = sorted(pkeys & ckeys)

    changed = []
    for key in common:
        a = prev[key]
        b = cur[key]
        if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
            changed.append(key)
        elif not torch.equal(a, b):
            changed.append(key)

    return {
        "previous": len(pkeys),
        "current": len(ckeys),
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


groups = [
    "policy_backend.vlm.",
    "policy_backend.act_query",
    "policy_backend.flow_action_query",
    "policy_backend.vlm_to_lam.",
    "policy_backend.lam.decoder.",
    "policy_backend.flow.",
]

print()
print(f"[freeze-check] {stage}: checkpoint parameter comparison")
for group in groups:
    r = compare(group)
    print(
        f"  {group:<38s} "
        f"prev={r['previous']:4d} cur={r['current']:4d} "
        f"changed={len(r['changed']):4d} "
        f"missing={len(r['missing']):3d} extra={len(r['extra']):3d}"
    )

vlm = compare("policy_backend.vlm.")
if (
    not vlm["previous"]
    or not vlm["current"]
    or vlm["missing"]
    or vlm["extra"]
    or vlm["changed"]
):
    examples = (
        vlm["missing"][:4]
        + vlm["extra"][:4]
        + vlm["changed"][:8]
    )
    raise RuntimeError(
        "Strict Freeze-VLM failed: policy_backend.vlm changed between "
        f"checkpoints at {stage}. Examples: {examples}"
    )

print(f"[OK] {stage}: policy_backend.vlm is bitwise unchanged.")

del prev, cur
gc.collect()
PY
}


# ==========================================================
# 11. Master log
# ==========================================================

MASTER_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${RUN_ROOT}/freeze_vlm_full_pipeline_${MASTER_TIMESTAMP}.log"

echo "[INFO] Master log:"
echo "       ${MASTER_LOG}"

exec > >(tee -a "${MASTER_LOG}") 2>&1


echo
echo "=========================================================="
echo " LaWAM LIBERO-Object Strict Freeze-VLM CL"
echo "=========================================================="
echo "Base checkpoint:"
echo "  ${BASE_RUN}"
echo
echo "CL checkpoint root:"
echo "  ${RUN_ROOT}"
echo
echo "Protocol:"
echo "  Base : reuse object 0-5 SeqFT Base"
echo "  CL1  : task 6, 2K"
echo "  CL2  : task 7, 2K"
echo "  CL3  : task 8, 2K"
echo "  CL4  : task 9, 2K"
echo
echo "Freeze policy during CL:"
echo "  Qwen vision backbone : FROZEN"
echo "  Qwen vision merger   : FROZEN"
echo "  Qwen LLM backbone    : FROZEN"
echo "  Qwen embeddings      : FROZEN"
echo "  act_query            : TRAINABLE"
echo "  flow_action_query    : TRAINABLE"
echo "  VLMToLAM / QFormer   : TRAINABLE"
echo "  LaWM decoder         : TRAINABLE"
echo "  Flow head            : TRAINABLE"
echo
echo "Training GPUs          : ${TRAIN_GPUS}"
echo "Batch / GPU            : ${PER_DEVICE_BATCH_SIZE}"
echo "Gradient accum.        : ${GRADIENT_ACCUMULATION_STEPS}"
echo "Global batch           : $((PER_DEVICE_BATCH_SIZE * 4 * GRADIENT_ACCUMULATION_STEPS))"
echo "Steps / CL stage       : ${MAX_TRAIN_STEPS}"
echo
echo "Evaluation policy GPU  : ${POLICY_GPU}"
echo "Evaluation sim GPU     : ${EVAL_GPU}"
echo "Evaluation workers     : ${EVAL_WORKERS}"
echo "Trials / task          : ${NUM_TRIALS}"
echo "Save videos            : ${SAVE_VIDEOS}"
echo
echo "START_FROM             : ${START_FROM}"
echo "CL_START_STAGE         : ${CL_START_STAGE}"
echo "=========================================================="
echo


# ==========================================================
# 12. Sequential Freeze-VLM CL training
# ==========================================================

if [ "${START_FROM}" = "cl" ]; then

    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES=4

    if [ "${CL_START_STAGE}" -eq 1 ]; then
        PREV_RUN="${BASE_RUN}"
    else
        PREV_STAGE=$((CL_START_STAGE - 1))
        PREV_TASK_ID=$((PREV_STAGE + 5))
        PREV_RUN_ID="cl${PREV_STAGE}_t${PREV_TASK_ID}_2k_4gpu_bs32_ga2_freeze_vlm"

        PREV_RUN=$(find_run \
            "${RUN_ROOT}" \
            "*+${PREV_RUN_ID}"
        )

        verify_run "Existing Freeze-VLM CL${PREV_STAGE}" "${PREV_RUN}"
        verify_freeze_config "${PREV_RUN}/config.yaml"
    fi

    for STAGE in $(seq "${CL_START_STAGE}" 4); do

        TASK_ID=$((STAGE + 5))
        RUN_ID="cl${STAGE}_t${TASK_ID}_2k_4gpu_bs32_ga2_freeze_vlm"

        PREV_CKPT="${PREV_RUN}/final_model/pytorch_model.pt"
        PREV_STATS="${PREV_RUN}/dataset_statistics.json"

        if [ ! -f "${PREV_CKPT}" ]; then
            echo "[ERROR] Previous checkpoint missing:"
            echo "        ${PREV_CKPT}"
            exit 1
        fi

        if [ ! -f "${PREV_STATS}" ]; then
            echo "[ERROR] Previous statistics missing:"
            echo "        ${PREV_STATS}"
            exit 1
        fi

        echo
        echo "=========================================================="
        echo " Starting Freeze-VLM CL${STAGE}"
        echo "=========================================================="
        echo "Task           : object ${TASK_ID}"
        echo "Previous run   : ${PREV_RUN}"
        echo "Previous ckpt  : ${PREV_CKPT}"
        echo "New run ID     : ${RUN_ID}"
        echo "=========================================================="
        echo

        bash train_lawam.sh \
            --run_root_dir="${RUN_ROOT}" \
            --run_id="${RUN_ID}" \
            \
            --datasets.vla_data.cl_suite=libero_object \
            "--datasets.vla_data.cl_task_ids=[${TASK_ID}]" \
            \
            --datasets.vla_data.use_task_filtered_statistics=false \
            --trainer.use_pretrained_dataset_statistics=true \
            --trainer.pretrained_checkpoint="${PREV_CKPT}" \
            \
            --trainer.freeze.freeze_vision_backbone=true \
            --trainer.freeze.freeze_llm_backbone=true \
            --trainer.freeze.freeze_last_llm_layer=true \
            --trainer.freeze.freeze_embedding=true \
            --trainer.freeze.unfreeze_vision_merger=false \
            --trainer.freeze.keep_llm_first_n_layers=16 \
            --trainer.freeze.unfreeze_llm_last_n_layers=-1 \
            --trainer.freeze.unfreeze_lam_decoder=true \
            \
            --datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
            --datasets.vla_data.num_workers="${NUM_WORKERS}" \
            --datasets.vla_data.val_num_workers="${VAL_NUM_WORKERS}" \
            --datasets.vla_data.persistent_workers=true \
            \
            --trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
            --trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
            --trainer.num_warmup_steps="${NUM_WARMUP_STEPS}" \
            \
            --trainer.logging_frequency="${LOGGING_FREQUENCY}" \
            --trainer.eval_interval="${TRAIN_EVAL_INTERVAL}" \
            --trainer.eval_batches="${TRAIN_EVAL_BATCHES}" \
            --trainer.save_interval="${SAVE_INTERVAL}"

        CURRENT_RUN=$(find_run \
            "${RUN_ROOT}" \
            "*+${RUN_ID}"
        )

        verify_run "Freeze-VLM CL${STAGE}" "${CURRENT_RUN}"

        echo
        echo "[INFO] Verifying fixed Base normalization after CL${STAGE}..."
        verify_statistics \
            "${BASE_STATS}" \
            "${CURRENT_RUN}/dataset_statistics.json"

        echo
        echo "[INFO] Verifying saved freeze policy after CL${STAGE}..."
        verify_freeze_config "${CURRENT_RUN}/config.yaml"

        echo
        echo "[INFO] Verifying checkpoint-level VLM invariance after CL${STAGE}..."
        verify_checkpoint_freeze \
            "${PREV_CKPT}" \
            "${CURRENT_RUN}/final_model/pytorch_model.pt" \
            "CL${STAGE}"

        echo
        echo "[OK] Freeze-VLM CL${STAGE} completed and verified."
        echo "     ${CURRENT_RUN}"
        echo

        PREV_RUN="${CURRENT_RUN}"
    done

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
fi


# ==========================================================
# 13. Resolve complete checkpoint chain
# ==========================================================

CL1_RUN=$(find_run \
    "${RUN_ROOT}" \
    '*+cl1_t6_2k_4gpu_bs32_ga2_freeze_vlm'
)

CL2_RUN=$(find_run \
    "${RUN_ROOT}" \
    '*+cl2_t7_2k_4gpu_bs32_ga2_freeze_vlm'
)

CL3_RUN=$(find_run \
    "${RUN_ROOT}" \
    '*+cl3_t8_2k_4gpu_bs32_ga2_freeze_vlm'
)

CL4_RUN=$(find_run \
    "${RUN_ROOT}" \
    '*+cl4_t9_2k_4gpu_bs32_ga2_freeze_vlm'
)

verify_run "Base" "${BASE_RUN}"
verify_run "Freeze-VLM CL1" "${CL1_RUN}"
verify_run "Freeze-VLM CL2" "${CL2_RUN}"
verify_run "Freeze-VLM CL3" "${CL3_RUN}"
verify_run "Freeze-VLM CL4" "${CL4_RUN}"

for RUN in "${CL1_RUN}" "${CL2_RUN}" "${CL3_RUN}" "${CL4_RUN}"; do
    verify_statistics "${BASE_STATS}" "${RUN}/dataset_statistics.json"
    verify_freeze_config "${RUN}/config.yaml"
done

# Strong final invariant: Base VLM must equal every CL-stage VLM.
for ITEM in \
    "CL1:${CL1_RUN}" \
    "CL2:${CL2_RUN}" \
    "CL3:${CL3_RUN}" \
    "CL4:${CL4_RUN}"
do
    STAGE_NAME="${ITEM%%:*}"
    RUN_PATH="${ITEM#*:}"
    verify_checkpoint_freeze \
        "${BASE_CKPT}" \
        "${RUN_PATH}/final_model/pytorch_model.pt" \
        "${STAGE_NAME}_vs_Base"
done


# ==========================================================
# 14. Evaluation environment
# ==========================================================

unset CUDA_VISIBLE_DEVICES || true
unset NUM_PROCESSES || true

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python


# ==========================================================
# 15. Evaluation master directory
# ==========================================================

EVAL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_DIR="${OUTPUT_ROOT}/cl_full_${EVAL_TIMESTAMP}"

mkdir -p "${MASTER_DIR}"

MANIFEST="${MASTER_DIR}/eval_manifest.tsv"

printf \
    "stage\tcheckpoint_run\teval_run_dir\ttask_ids\n" \
    > "${MANIFEST}"


# ==========================================================
# 16. Helper: evaluate one stage
# ==========================================================

run_eval_stage() {

    local stage="$1"
    local train_run="$2"
    local task_ids="$3"
    local alias="$4"

    local ckpt="${train_run}/final_model/pytorch_model.pt"

    echo
    echo "=========================================================="
    echo " Evaluating ${stage}"
    echo "=========================================================="
    echo "Checkpoint:"
    echo "  ${ckpt}"
    echo "Task IDs:"
    echo "  ${task_ids}"
    echo "=========================================================="
    echo

    SUITES="libero_object" \
    TASK_IDS="${task_ids}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${EVAL_WORKERS}" \
    GPU_IDS="${POLICY_GPU}" \
    EVAL_GPU_IDS="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${MASTER_DIR}" \
    LIBERO_CKPT_ALIAS="${alias}" \
    bash \
    examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh \
    "${ckpt}"

    local eval_timestamp_dir
    eval_timestamp_dir=$(find \
        "${MASTER_DIR}/${alias}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        | sort \
        | tail -n 1
    )

    if [ -z "${eval_timestamp_dir}" ]; then
        echo "[ERROR] Evaluation output not found:"
        echo "        ${MASTER_DIR}/${alias}"
        exit 1
    fi

    local suite_dir="${eval_timestamp_dir}/suites/libero_object"

    if [ ! -f "${suite_dir}/episodes.jsonl" ]; then
        echo "[ERROR] episodes.jsonl not found:"
        echo "        ${suite_dir}/episodes.jsonl"
        exit 1
    fi

    if [ ! -f "${suite_dir}/summary.json" ]; then
        echo "[ERROR] summary.json not found:"
        echo "        ${suite_dir}/summary.json"
        exit 1
    fi

    read -r -a TASK_ARRAY <<< "${task_ids}"

    echo
    echo "[INFO] Aggregating ${stage} per-task results..."

    python "${SUMMARY_SCRIPT}" \
        --run-dir "${suite_dir}" \
        --task-ids "${TASK_ARRAY[@]}" \
        --expected-trials "${NUM_TRIALS}"

    printf \
        "%s\t%s\t%s\t%s\n" \
        "${stage}" \
        "${train_run}" \
        "${suite_dir}" \
        "${task_ids}" \
        >> "${MANIFEST}"

    echo
    echo "[OK] ${stage} evaluation completed."
    echo "     ${suite_dir}"
}


# ==========================================================
# 17. Full sequential evaluation
# ==========================================================

echo
echo "=========================================================="
echo " Starting full Freeze-VLM CL evaluation"
echo "=========================================================="
echo "Policy GPU       : ${POLICY_GPU}"
echo "Evaluator GPU    : ${EVAL_GPU}"
echo "Workers          : ${EVAL_WORKERS}"
echo "Trials / task    : ${NUM_TRIALS}"
echo "Save videos      : ${SAVE_VIDEOS}"
echo "Output           : ${MASTER_DIR}"
echo "=========================================================="

run_eval_stage \
    "Base" \
    "${BASE_RUN}" \
    "0 1 2 3 4 5" \
    "base_t0_5_10k"

run_eval_stage \
    "CL1" \
    "${CL1_RUN}" \
    "0 1 2 3 4 5 6" \
    "freeze_vlm_cl1_t6_2k"

run_eval_stage \
    "CL2" \
    "${CL2_RUN}" \
    "0 1 2 3 4 5 6 7" \
    "freeze_vlm_cl2_t7_2k"

run_eval_stage \
    "CL3" \
    "${CL3_RUN}" \
    "0 1 2 3 4 5 6 7 8" \
    "freeze_vlm_cl3_t8_2k"

run_eval_stage \
    "CL4" \
    "${CL4_RUN}" \
    "0 1 2 3 4 5 6 7 8 9" \
    "freeze_vlm_cl4_t9_2k"


# ==========================================================
# 18. Build CL performance matrix
# ==========================================================

echo
echo "=========================================================="
echo " Building Freeze-VLM CL performance matrix"
echo "=========================================================="

python - "${MANIFEST}" "${MASTER_DIR}" "${NUM_TRIALS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
master_dir = Path(sys.argv[2])
expected_trials = int(sys.argv[3])

with manifest_path.open("r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f, delimiter="\t"))

long_rows = []
stage_results = {}
stage_summaries = []

for item in manifest_rows:
    stage = item["stage"]
    run_dir = Path(item["eval_run_dir"])
    per_task_path = run_dir / "per_task_summary.json"

    with per_task_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    stage_results[stage] = {}

    for row in rows:
        task_id = int(row["task_id"])
        sr = float(row["success_rate"])
        successes = int(row["successes"])
        trials = int(row["trials"])

        if trials != expected_trials:
            raise RuntimeError(
                f"{stage}/task{task_id}: expected {expected_trials} trials, got {trials}"
            )

        stage_results[stage][task_id] = sr

        long_rows.append(
            {
                "stage": stage,
                "task_id": task_id,
                "task_description": row["task_description"],
                "successes": successes,
                "trials": trials,
                "success_rate": sr,
            }
        )

    total_success = sum(int(row["successes"]) for row in rows)
    total_trials = sum(int(row["trials"]) for row in rows)
    mean_task_sr = sum(float(row["success_rate"]) for row in rows) / len(rows)

    stage_summaries.append(
        {
            "stage": stage,
            "num_tasks": len(rows),
            "successes": total_success,
            "trials": total_trials,
            "overall_success_rate": total_success / total_trials,
            "mean_task_success_rate": mean_task_sr,
        }
    )


long_path = master_dir / "cl_per_task_long.csv"
with long_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stage",
            "task_id",
            "task_description",
            "successes",
            "trials",
            "success_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(long_rows)


stage_summary_path = master_dir / "cl_stage_summary.csv"
with stage_summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "stage",
            "num_tasks",
            "successes",
            "trials",
            "overall_success_rate",
            "mean_task_success_rate",
        ],
    )
    writer.writeheader()
    writer.writerows(stage_summaries)


stage_order = ["Base", "CL1", "CL2", "CL3", "CL4"]

matrix_path = master_dir / "cl_performance_matrix.csv"
with matrix_path.open("w", newline="", encoding="utf-8") as f:
    fieldnames = (
        ["stage"]
        + [f"task_{task_id}" for task_id in range(10)]
        + ["mean_seen_task_sr"]
    )

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for stage in stage_order:
        result = stage_results[stage]
        row = {"stage": stage}

        for task_id in range(10):
            row[f"task_{task_id}"] = (
                f"{result[task_id]:.4f}"
                if task_id in result
                else ""
            )

        row["mean_seen_task_sr"] = (
            f"{sum(result.values()) / len(result):.4f}"
        )
        writer.writerow(row)


print()
print("Stage | " + " | ".join(f"T{i}" for i in range(10)) + " | Mean")
print("-" * 104)

for stage in stage_order:
    result = stage_results[stage]
    values = [
        f"{result[i]:.2f}" if i in result else "-"
        for i in range(10)
    ]
    mean_sr = sum(result.values()) / len(result)

    print(
        f"{stage:>4s} | "
        + " | ".join(f"{value:>4s}" for value in values)
        + f" | {mean_sr:.4f}"
    )

print()
print("[OK] Saved:")
print(matrix_path)
print(long_path)
print(stage_summary_path)
PY


# ==========================================================
# 19. Final summary
# ==========================================================

echo
echo "=========================================================="
echo " LIBERO-Object Strict Freeze-VLM experiment completed"
echo "=========================================================="
echo "Base (reused):"
echo "  ${BASE_RUN}"
echo
echo "Freeze-VLM chain:"
echo "  CL1: ${CL1_RUN}"
echo "  CL2: ${CL2_RUN}"
echo "  CL3: ${CL3_RUN}"
echo "  CL4: ${CL4_RUN}"
echo
echo "Evaluation directory:"
echo "  ${MASTER_DIR}"
echo
echo "Performance matrix:"
echo "  ${MASTER_DIR}/cl_performance_matrix.csv"
echo
echo "Per-task long table:"
echo "  ${MASTER_DIR}/cl_per_task_long.csv"
echo
echo "Stage summary:"
echo "  ${MASTER_DIR}/cl_stage_summary.csv"
echo
echo "Manifest:"
echo "  ${MANIFEST}"
echo
echo "Master log:"
echo "  ${MASTER_LOG}"
echo "=========================================================="