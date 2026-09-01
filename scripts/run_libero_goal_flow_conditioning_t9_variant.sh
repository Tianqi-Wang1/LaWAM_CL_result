#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# One fresh T9 conditioning experiment.
#
# MODE:
#   reference : fresh Level-2 r8 rerun
#   adanorm   : Level-2 r8 + 16 x norm1/AdaNorm Linear LoRA
#   timestep  : Level-2 r8 + timestep_encoder Linear LoRA
#   both      : Level-2 r8 + AdaNorm + timestep LoRA
#
# Every run starts independently from the same Formal Base.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
MERGE_SCRIPT="${ROOT}/scripts/merge_flow_conditioning_lora_checkpoint.py"

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

MODE="${MODE:-reference}"

case "${MODE}" in
    reference)
        TARGET_ADANORM=false
        TARGET_TIMESTEP=false
        TARGET_TAG=fresh_level2_r8_reference
        EXPECTED_MODULES=99
        EXPECTED_PARAMS=2390016
        ;;
    adanorm)
        TARGET_ADANORM=true
        TARGET_TIMESTEP=false
        TARGET_TAG=level2_r8_plus_adanorm
        EXPECTED_MODULES=115
        EXPECTED_PARAMS=2783232
        ;;
    timestep)
        TARGET_ADANORM=false
        TARGET_TIMESTEP=true
        TARGET_TAG=level2_r8_plus_timestep
        EXPECTED_MODULES=101
        EXPECTED_PARAMS=2416640
        ;;
    both)
        TARGET_ADANORM=true
        TARGET_TIMESTEP=true
        TARGET_TAG=conditioning_complete_r8
        EXPECTED_MODULES=117
        EXPECTED_PARAMS=2809856
        ;;
    *)
        echo "[ERROR] MODE must be: reference | adanorm | timestep | both"
        exit 1
        ;;
esac

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LR="${LORA_LR:-0.0001}"

if [ "${LORA_RANK}" != "8" ] || [ "${LORA_ALPHA}" != "8" ]; then
    echo "[ERROR] This controlled experiment is fixed to rank=8, alpha=8."
    exit 1
fi

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

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_conditioning_complete_t9/${TARGET_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_conditioning_complete_t9/${TARGET_TAG}"

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

# Code support preflight.
python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)
fields=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required={
    "train_flow_lora",
    "flow_lora_target_adanorm",
    "flow_lora_target_timestep",
}
missing=sorted(required-fields)
if missing:
    raise RuntimeError(
        f"Conditioning-LoRA support missing: {missing}"
    )
print("[OK] Conditioning-LoRA code support detected.")
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
echo " LaWAM Goal T9 Conditioning-LoRA"
echo "=========================================================="
echo "Mode           : ${MODE}"
echo "Base           : ${BASE_RUN}"
echo "Level-2 core   : ON"
echo "AdaNorm LoRA   : ${TARGET_ADANORM}"
echo "Timestep LoRA  : ${TARGET_TIMESTEP}"
echo "rank/alpha     : ${LORA_RANK}/${LORA_ALPHA}"
echo "LR             : ${LORA_LR}"
echo "Expected mods  : ${EXPECTED_MODULES}"
echo "Expected params: ${EXPECTED_PARAMS}"
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
    "--trainer.freeze.flow_lora_target_adanorm=${TARGET_ADANORM}" \
    "--trainer.freeze.flow_lora_target_timestep=${TARGET_TIMESTEP}" \
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
ADAPTER="${RUN}/final_model/flow_lora_adapter.pt"

# Normalization audit.
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
    "${EXPECTED_MODULES}" \
    "${EXPECTED_PARAMS}" <<'PY'
import re,sys,torch

bp,cp,mode,rank_s,module_s,param_s=sys.argv[1:7]
rank=int(rank_s)
expected_modules=int(module_s)
expected_params=int(param_s)

def load(p):
    try:return torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try:return torch.load(p,map_location="cpu",weights_only=True)
        except TypeError:return torch.load(p,map_location="cpu")

base=load(bp)
cur=load(cp)

bk={k for k,v in base.items() if torch.is_tensor(v)}
ck={k for k,v in cur.items() if torch.is_tensor(v)}

missing=sorted(bk-ck)
if missing:
    raise RuntimeError(f"Missing Base tensors: {missing[:30]}")

changed=[]
for k in sorted(bk):
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
    if not (k.endswith(".lora_A") or k.endswith(".lora_B"))
]
if bad:
    raise RuntimeError(f"Unexpected extra tensors: {bad[:30]}")

canon_A=sorted(
    k for k in extra
    if k.startswith("policy_backend.flow.") and k.endswith(".lora_A")
)
canon_B=sorted(
    k for k in extra
    if k.startswith("policy_backend.flow.") and k.endswith(".lora_B")
)
alias_A=sorted(
    k for k in extra
    if k.startswith("policy_action_head.") and k.endswith(".lora_A")
)
alias_B=sorted(
    k for k in extra
    if k.startswith("policy_action_head.") and k.endswith(".lora_B")
)

if len(canon_A)!=expected_modules or len(canon_B)!=expected_modules:
    raise RuntimeError(
        f"Canonical module count expected={expected_modules}, "
        f"A={len(canon_A)}, B={len(canon_B)}"
    )

if len(alias_A)!=len(canon_A) or len(alias_B)!=len(canon_B):
    raise RuntimeError("Canonical/alias LoRA count mismatch")

def prefix(k):
    return k[len("policy_backend.flow."):-len(".lora_A")]

groups={
    "attention":0,
    "ffn":0,
    "enc_vlm":0,
    "output":0,
    "adanorm":0,
    "timestep":0,
}

unexpected=[]

for k in canon_A:
    p=prefix(k)

    if re.match(r"^DiT\.transformer_blocks\.\d+\.attn1\.",p):
        groups["attention"]+=1
    elif re.match(r"^DiT\.transformer_blocks\.\d+\.ff\.",p):
        groups["ffn"]+=1
    elif p.startswith("enc_vlm."):
        groups["enc_vlm"]+=1
    elif p in ("DiT.proj_out_1","DiT.proj_out_2"):
        groups["output"]+=1
    elif re.match(r"^DiT\.transformer_blocks\.\d+\.norm1\.",p):
        groups["adanorm"]+=1
    elif p.startswith("DiT.timestep_encoder."):
        groups["timestep"]+=1
    else:
        unexpected.append(p)

if unexpected:
    raise RuntimeError(
        f"LoRA outside requested target set: {unexpected[:30]}"
    )

expected_groups={
    "attention":64,
    "ffn":32,
    "enc_vlm":1,
    "output":2,
    "adanorm":16 if mode in ("adanorm","both") else 0,
    "timestep":2 if mode in ("timestep","both") else 0,
}

for g,e in expected_groups.items():
    if groups[g]!=e:
        raise RuntimeError(
            f"{g}: expected {e}, got {groups[g]}"
        )

for k in canon_A:
    if int(cur[k].shape[0]) != rank:
        raise RuntimeError(
            f"Rank mismatch: {k} {tuple(cur[k].shape)}"
        )

actual_params=sum(cur[k].numel() for k in canon_A+canon_B)
if actual_params!=expected_params:
    raise RuntimeError(
        f"Params expected={expected_params:,}, got={actual_params:,}"
    )

B_abs=sum(
    float(cur[k].float().abs().sum().item())
    for k in canon_B
)
if B_abs<=0:
    raise RuntimeError("LoRA B tensors did not train")

print(
    "[conditioning-lora-check] "
    f"mode={mode}, base_checked={len(bk)}, base_changed=0, "
    f"canonical_modules={len(canon_A)}, "
    f"adapter_params={actual_params:,}, "
    f"group_modules={groups}, "
    f"B_abs_sum={B_abs:.6g}, exact_base=True"
)
PY

# Merge to standard LaWAM checkpoint.
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

base=load(bp)
unmerged=load(up)
merged=load(mp)

bk={k for k,v in base.items() if torch.is_tensor(v)}
mk={k for k,v in merged.items() if torch.is_tensor(v)}

if bk != mk:
    raise RuntimeError(
        f"Merged key mismatch missing={sorted(bk-mk)[:20]} "
        f"extra={sorted(mk-bk)[:20]}"
    )

allowed={
    k[:-len(".lora_A")]+".weight"
    for k,v in unmerged.items()
    if torch.is_tensor(v) and k.endswith(".lora_A")
}

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
        f"Merged changed non-target tensors: {bad[:30]}"
    )

if not changed:
    raise RuntimeError("Merged checkpoint has no changed targets")

print(
    "[conditioning-lora-merged-check] "
    f"changed_target_tensors={len(changed)}, "
    "changed_non_target=0, standard_checkpoint_keys=True"
)
PY

# T9 evaluation.
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
echo " T9 conditioning experiment complete"
echo "=========================================================="
echo "Mode    : ${MODE}"
echo "Run     : ${RUN}"
echo "Adapter : ${ADAPTER}"
echo "Merged  : ${MERGED}"
echo "Result  : ${RESULT}"
echo "Log     : ${MASTER_LOG}"
echo "=========================================================="
