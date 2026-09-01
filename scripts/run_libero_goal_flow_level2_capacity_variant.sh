#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# One T9 capacity experiment.
#
# MODE=broad_r32:
#   Transformer + enc_vlm + output -> LoRA r32/a32
#
# MODE=dense_interface:
#   Transformer -> LoRA r8/a8
#   enc_vlm + output -> zero-init full-rank delta
#
# All original Base WAM/Flow tensors remain frozen.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
MERGE_SCRIPT="${ROOT}/scripts/merge_flow_level2_capacity_pair.py"

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

MODE="${MODE:-broad_r32}"

case "${MODE}" in
    broad_r32)
        LORA_RANK=32
        LORA_ALPHA=32
        INTERFACE_MODE=lora
        EXPECTED_CANONICAL_LORA_MODULES=99
        EXPECTED_CANONICAL_DENSE_MODULES=0
        EXPECTED_CANONICAL_PARAMS=9560064
        TARGET_TAG=level2_broad_r32
        ;;
    dense_interface)
        LORA_RANK=8
        LORA_ALPHA=8
        INTERFACE_MODE=dense
        EXPECTED_CANONICAL_LORA_MODULES=96
        EXPECTED_CANONICAL_DENSE_MODULES=3
        EXPECTED_CANONICAL_PARAMS=7045120
        TARGET_TAG=transformer_r8_dense_interfaces
        ;;
    *)
        echo "[ERROR] MODE must be broad_r32 or dense_interface"
        exit 1
        ;;
esac

LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LR="${LORA_LR:-0.0001}"

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

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_level2_capacity_pair/${TARGET_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_level2_capacity_pair/${TARGET_TAG}"

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
}

python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)
fields=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required={"train_flow_lora","flow_interface_adapter_mode"}
missing=sorted(required-fields)
if missing:
    raise RuntimeError(f"Capacity-pair support missing: {missing}")
print("[OK] Level-2 capacity-pair support detected.")
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

RUN_ID="t9_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_${TARGET_TAG}_from_base"

TS=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/t9_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " LaWAM Goal T9 Level-2 Capacity Experiment"
echo "=========================================================="
echo "Mode           : ${MODE}"
echo "Base           : ${BASE_RUN}"
echo "Transformer    : LoRA r${LORA_RANK}/a${LORA_ALPHA}"
echo "Interface mode : ${INTERFACE_MODE}"
echo "LR             : ${LORA_LR}"
echo "Expected params: ${EXPECTED_CANONICAL_PARAMS}"
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
    "--trainer.freeze.flow_interface_adapter_mode=${INTERFACE_MODE}" \
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
verify_run "${MODE}" "${RUN}"

UNMERGED="${RUN}/final_model/pytorch_model.pt"
MERGED="${RUN}/final_model/pytorch_model_merged.pt"
ADAPTER="${RUN}/final_model/flow_task_adapter.pt"

# Normalization.
python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1],"r",encoding="utf-8"))
b=json.load(open(sys.argv[2],"r",encoding="utf-8"))
for tag in a:
    for sec in ("action","state"):
        if a[tag][sec] != b[tag][sec]:
            raise RuntimeError(f"Normalization changed: {tag}/{sec}")
print("[OK] action/state normalization identical to Base.")
PY

# HARD unmerged audit.
python - \
    "${BASE_CKPT}" \
    "${UNMERGED}" \
    "${MODE}" \
    "${LORA_RANK}" \
    "${EXPECTED_CANONICAL_LORA_MODULES}" \
    "${EXPECTED_CANONICAL_DENSE_MODULES}" \
    "${EXPECTED_CANONICAL_PARAMS}" <<'PY'
import re,sys,torch

bp,cp,mode,rank_s,lora_m_s,dense_m_s,param_s=sys.argv[1:8]
rank=int(rank_s)
expected_lora=int(lora_m_s)
expected_dense=int(dense_m_s)
expected_params=int(param_s)

def load(p):
    try:return torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try:return torch.load(p,map_location="cpu",weights_only=True)
        except TypeError:return torch.load(p,map_location="cpu")

base=load(bp); cur=load(cp)

bk={k for k,v in base.items() if torch.is_tensor(v)}
ck={k for k,v in cur.items() if torch.is_tensor(v)}

if bk-ck:
    raise RuntimeError(f"Missing Base keys: {sorted(bk-ck)[:20]}")

changed=[]
for k in bk:
    a,b=base[k],cur[k]
    if (
        tuple(a.shape)!=tuple(b.shape)
        or a.dtype!=b.dtype
        or not torch.equal(a,b)
    ):
        changed.append(k)

if changed:
    raise RuntimeError(
        f"BASE FREEZE FAILED: {len(changed)} changed: {changed[:30]}"
    )

extra=sorted(ck-bk)
bad=[
    k for k in extra
    if not (
        k.endswith(".lora_A")
        or k.endswith(".lora_B")
        or k.endswith(".delta_weight")
    )
]
if bad:
    raise RuntimeError(f"Unexpected extra keys: {bad[:30]}")

canon_A=sorted(
    k for k in extra
    if k.startswith("policy_backend.flow.") and k.endswith(".lora_A")
)
canon_B=sorted(
    k for k in extra
    if k.startswith("policy_backend.flow.") and k.endswith(".lora_B")
)
canon_D=sorted(
    k for k in extra
    if k.startswith("policy_backend.flow.") and k.endswith(".delta_weight")
)

alias_A=sorted(
    k for k in extra
    if k.startswith("policy_action_head.") and k.endswith(".lora_A")
)
alias_B=sorted(
    k for k in extra
    if k.startswith("policy_action_head.") and k.endswith(".lora_B")
)
alias_D=sorted(
    k for k in extra
    if k.startswith("policy_action_head.") and k.endswith(".delta_weight")
)

if len(canon_A)!=expected_lora or len(canon_B)!=expected_lora:
    raise RuntimeError(
        f"Canonical LoRA count expected {expected_lora}, "
        f"got A={len(canon_A)}, B={len(canon_B)}"
    )
if len(canon_D)!=expected_dense:
    raise RuntimeError(
        f"Canonical dense count expected {expected_dense}, got {len(canon_D)}"
    )
if len(alias_A)!=len(canon_A) or len(alias_B)!=len(canon_B):
    raise RuntimeError("LoRA alias count mismatch")
if len(alias_D)!=len(canon_D):
    raise RuntimeError("Dense alias count mismatch")

# Group audit.
def prefix_of(k, suffix):
    return k[len("policy_backend.flow."):-len(suffix)]

lora_prefixes=[prefix_of(k,".lora_A") for k in canon_A]
dense_prefixes=[prefix_of(k,".delta_weight") for k in canon_D]

def is_transformer(p):
    return (
        re.match(r"^DiT\.transformer_blocks\.\d+\.attn1\.",p)
        or re.match(r"^DiT\.transformer_blocks\.\d+\.ff\.",p)
    )

t_lora=sum(bool(is_transformer(p)) for p in lora_prefixes)
i_lora=sum(
    p.startswith("enc_vlm.")
    or p in ("DiT.proj_out_1","DiT.proj_out_2")
    for p in lora_prefixes
)
i_dense=sum(
    p.startswith("enc_vlm.")
    or p in ("DiT.proj_out_1","DiT.proj_out_2")
    for p in dense_prefixes
)

if t_lora != 96:
    raise RuntimeError(f"Expected 96 Transformer LoRA modules, got {t_lora}")

if mode=="broad_r32":
    if i_lora!=3 or i_dense!=0:
        raise RuntimeError(
            f"broad_r32 interface groups wrong: lora={i_lora}, dense={i_dense}"
        )
else:
    if i_lora!=0 or i_dense!=3:
        raise RuntimeError(
            f"dense_interface groups wrong: lora={i_lora}, dense={i_dense}"
        )

# Shape/rank and trainable parameter count.
for k in canon_A:
    if int(cur[k].shape[0]) != rank:
        raise RuntimeError(f"Rank mismatch: {k} {tuple(cur[k].shape)}")

actual_params=sum(cur[k].numel() for k in canon_A+canon_B+canon_D)
if actual_params != expected_params:
    raise RuntimeError(
        f"Adapter params expected {expected_params:,}, got {actual_params:,}"
    )

B_abs=sum(float(cur[k].float().abs().sum().item()) for k in canon_B)
D_abs=sum(float(cur[k].float().abs().sum().item()) for k in canon_D)

if B_abs <= 0:
    raise RuntimeError("LoRA B did not train")
if expected_dense and D_abs <= 0:
    raise RuntimeError("Dense interface delta did not train")

print(
    "[capacity-pair-check] "
    f"mode={mode}, base_checked={len(bk)}, base_changed=0, "
    f"lora_modules={len(canon_A)}, dense_modules={len(canon_D)}, "
    f"adapter_params={actual_params:,}, "
    f"B_abs_sum={B_abs:.6g}, D_abs_sum={D_abs:.6g}, "
    "exact_base=True"
)
PY

python "${MERGE_SCRIPT}" \
    "${UNMERGED}" \
    --output "${MERGED}" \
    --adapter-output "${ADAPTER}" \
    --alpha "${LORA_ALPHA}"

# HARD merged audit.
python - "${BASE_CKPT}" "${UNMERGED}" "${MERGED}" <<'PY'
import sys,torch

bp,up,mp=sys.argv[1:4]

def load(p):
    try:return torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try:return torch.load(p,map_location="cpu",weights_only=True)
        except TypeError:return torch.load(p,map_location="cpu")

base=load(bp); unmerged=load(up); merged=load(mp)

bk={k for k,v in base.items() if torch.is_tensor(v)}
mk={k for k,v in merged.items() if torch.is_tensor(v)}

if bk != mk:
    raise RuntimeError(
        f"Merged key mismatch missing={sorted(bk-mk)[:20]} "
        f"extra={sorted(mk-bk)[:20]}"
    )

allowed=set()

for k,v in unmerged.items():
    if not torch.is_tensor(v):
        continue
    if k.endswith(".lora_A"):
        allowed.add(k[:-len(".lora_A")]+".weight")
    elif k.endswith(".delta_weight"):
        allowed.add(k[:-len(".delta_weight")]+".weight")

changed=[]
bad=[]

for k in sorted(bk):
    a,b=base[k],merged[k]
    if (
        tuple(a.shape)!=tuple(b.shape)
        or a.dtype!=b.dtype
        or not torch.equal(a,b)
    ):
        changed.append(k)
        if k not in allowed:
            bad.append(k)

if bad:
    raise RuntimeError(
        f"Merged changed non-target Base tensors: {bad[:30]}"
    )
if not changed:
    raise RuntimeError("Merged checkpoint has no changed targets")

print(
    "[capacity-pair-merged-check] "
    f"changed_target_tensors={len(changed)}, "
    "changed_non_target=0, standard_checkpoint_keys=True"
)
PY

# Evaluate T9.
export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

EVAL_MASTER="${OUTPUT_ROOT}/t9_${TS}"
ALIAS="${TARGET_TAG}_T9"

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
    | sort | tail -n 1)

[ -n "${EVAL_DIR}" ] || {
    echo "[ERROR] Evaluation output not found"
    exit 1
}

SUITE_DIR="${EVAL_DIR}/suites/libero_goal"

python "${SUMMARY_SCRIPT}" \
    --run-dir "${SUITE_DIR}" \
    --task-ids 9 \
    --expected-trials "${NUM_TRIALS}"

RESULT="${SUITE_DIR}/per_task_summary.csv"

echo "${RESULT}" > "${EXPERIMENT_ROOT}/LATEST_RESULT.txt"
echo "${RUN}" > "${EXPERIMENT_ROOT}/LATEST_RUN.txt"

echo
echo "=========================================================="
echo " T9 capacity experiment complete"
echo "=========================================================="
echo "Mode    : ${MODE}"
echo "Run     : ${RUN}"
echo "Adapter : ${ADAPTER}"
echo "Merged  : ${MERGED}"
echo "Result  : ${RESULT}"
echo "Log     : ${MASTER_LOG}"
echo "=========================================================="
