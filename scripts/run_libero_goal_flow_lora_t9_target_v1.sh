#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LaWAM LIBERO-Goal T9: Frozen Base WAM/Flow + Broad Flow-LoRA target diagnostic
#
# Transformer Q/K/V/O+FFN stays enabled by default. Optional additions:
#   flow.enc_vlm (all descendant nn.Linear)
#   flow.enc_wm  (all descendant nn.Linear)
#   flow.DiT.proj_out_1 / proj_out_2
#
# Every run starts from the SAME formal Base checkpoint.
# Only *.lora_A / *.lora_B are trainable.
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
MERGE_SCRIPT="${ROOT}/scripts/merge_flow_lora_checkpoint.py"
FREEZE_POLICY_PY="${ROOT}/starVLA/model/framework/latent_world/runtime/freeze_policy.py"
FLOW_LORA_PY="${ROOT}/starVLA/model/framework/latent_world/runtime/flow_lora.py"

for required in "${SUMMARY_SCRIPT}" "${MERGE_SCRIPT}" "${FREEZE_POLICY_PY}" "${FLOW_LORA_PY}"; do
    [ -f "${required}" ] || { echo "[ERROR] Missing ${required}"; exit 1; }
done

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_LR="${LORA_LR:-0.0001}"

LORA_TARGET_ATTN="${LORA_TARGET_ATTN:-true}"
LORA_TARGET_FFN="${LORA_TARGET_FFN:-true}"
LORA_TARGET_ENC_VLM="${LORA_TARGET_ENC_VLM:-false}"
LORA_TARGET_ENC_WM="${LORA_TARGET_ENC_WM:-false}"
LORA_TARGET_OUTPUT="${LORA_TARGET_OUTPUT:-false}"
TARGET_TAG="${TARGET_TAG:-custom}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
[ "${NUM_TRAIN_GPUS}" -ge 1 ] || { echo "[ERROR] No TRAIN_GPUS"; exit 1; }

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
FROZEN_GROUP_LR="${FROZEN_GROUP_LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"

NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

sanitize_tag() {
    echo "$1" | sed -e 's/\./p/g' -e 's/-/m/g' -e 's/+//g'
}
ALPHA_TAG=$(sanitize_tag "${LORA_ALPHA}")
LR_TAG=$(sanitize_tag "${LORA_LR}")
FLOW_LORA_TAG="r${LORA_RANK}_a${ALPHA_TAG}_lr${LR_TAG}_${TARGET_TAG}"

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_lora_target_ablation/${FLOW_LORA_TAG}"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_lora_target_ablation/${FLOW_LORA_TAG}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

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
    for p in "${run}/config.yaml" "${run}/dataset_statistics.json" "${run}/final_model/pytorch_model.pt"; do
        [ -f "${p}" ] || { echo "[ERROR] ${label}: missing ${p}"; exit 1; }
    done
    echo "[OK] ${label}: ${run}"
}

# Code-support preflight.
python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import LatentWorldPolicyFreezeConfig
fields=set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required={
    "train_flow_lora",
    "flow_lora_target_enc_vlm",
    "flow_lora_target_enc_wm",
    "flow_lora_target_output",
}
missing=sorted(required-fields)
if missing:
    raise RuntimeError(
        f"Broad Flow-LoRA support missing: {missing}. "
        "Run install_flow_lora_broad_v2.sh first."
    )
print("[OK] Broad Flow-LoRA support detected.")
PY

if [ -n "${BASE_RUN:-}" ]; then
    :
else
    BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

RUN_ID="t9_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_${FLOW_LORA_TAG}_from_base"
PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/t9_${PIPELINE_TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " Goal T9 Broad Flow-LoRA"
echo "=========================================================="
echo "Base           : ${BASE_RUN}"
echo "r/alpha/LR     : ${LORA_RANK}/${LORA_ALPHA}/${LORA_LR}"
echo "Attention      : ${LORA_TARGET_ATTN}"
echo "FFN            : ${LORA_TARGET_FFN}"
echo "enc_vlm        : ${LORA_TARGET_ENC_VLM}"
echo "enc_wm         : ${LORA_TARGET_ENC_WM}"
echo "output         : ${LORA_TARGET_OUTPUT}"
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
    "--trainer.freeze.flow_lora_target_attention=${LORA_TARGET_ATTN}" \
    "--trainer.freeze.flow_lora_target_ffn=${LORA_TARGET_FFN}" \
    "--trainer.freeze.flow_lora_target_enc_vlm=${LORA_TARGET_ENC_VLM}" \
    "--trainer.freeze.flow_lora_target_enc_wm=${LORA_TARGET_ENC_WM}" \
    "--trainer.freeze.flow_lora_target_output=${LORA_TARGET_OUTPUT}" \
    "--trainer.freeze.unfreeze_lam_decoder=false" \
    "--trainer.learning_rate.vlm.lr=${FROZEN_GROUP_LR}" \
    "--trainer.learning_rate.action_model.lr=${LORA_LR}" \
    "--trainer.learning_rate.world_model.lr=${FROZEN_GROUP_LR}" \
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

# Dataset statistics must stay identical to formal Base.
python - "${BASE_STATS}" "${RUN}/dataset_statistics.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1],"r",encoding="utf-8"))
b=json.load(open(sys.argv[2],"r",encoding="utf-8"))
for tag in a:
    if tag not in b:
        raise RuntimeError(f"Missing normalization tag: {tag}")
    for sec in ("action","state"):
        if a[tag][sec] != b[tag][sec]:
            raise RuntimeError(f"Normalization changed: {tag}/{sec}")
print("[OK] action/state normalization identical to Base.")
PY

# Saved config audit.
python - \
    "${RUN}/config.yaml" "${BASE_CKPT}" \
    "${LORA_RANK}" "${LORA_ALPHA}" "${LORA_DROPOUT}" "${LORA_LR}" \
    "${LORA_TARGET_ATTN}" "${LORA_TARGET_FFN}" \
    "${LORA_TARGET_ENC_VLM}" "${LORA_TARGET_ENC_WM}" "${LORA_TARGET_OUTPUT}" <<'PY'
import math,sys
from omegaconf import OmegaConf
(
 path,base,rank_s,alpha_s,drop_s,lr_s,
 attn_s,ffn_s,vlm_s,wm_s,out_s
)=sys.argv[1:12]
def as_bool(x): return str(x).lower()=="true"
cfg=OmegaConf.load(path); fr=cfg.trainer.freeze
expected={
 "train_flow_only":False,
 "train_flow_lora":True,
 "flow_lora_rank":int(rank_s),
 "flow_lora_alpha":float(alpha_s),
 "flow_lora_dropout":float(drop_s),
 "flow_lora_target_attention":as_bool(attn_s),
 "flow_lora_target_ffn":as_bool(ffn_s),
 "flow_lora_target_enc_vlm":as_bool(vlm_s),
 "flow_lora_target_enc_wm":as_bool(wm_s),
 "flow_lora_target_output":as_bool(out_s),
 "unfreeze_lam_decoder":False,
}
for k,v in expected.items():
    a=fr.get(k,None)
    if isinstance(v,float):
        if not math.isclose(float(a),v,rel_tol=0,abs_tol=1e-12):
            raise RuntimeError(f"{k}: {a} != {v}")
    elif a != v:
        raise RuntimeError(f"{k}: {a} != {v}")
if str(cfg.trainer.pretrained_checkpoint)!=str(base):
    raise RuntimeError("Not initialized from formal Base checkpoint")
if list(cfg.datasets.vla_data.cl_task_ids)!=[9]:
    raise RuntimeError("Wrong task filter; expected [9]")
if not bool(cfg.trainer.load_pretrained_policy_flow):
    raise RuntimeError("load_pretrained_policy_flow must be true")
if not math.isclose(float(cfg.trainer.learning_rate.action_model.lr),float(lr_s),rel_tol=0,abs_tol=1e-12):
    raise RuntimeError("Wrong action_model LR")
print("[OK] Broad Flow-LoRA config verified.")
PY

# HARD unmerged checkpoint audit:
#   - every original Base tensor is bitwise unchanged
#   - the only extra tensors are LoRA A/B
#   - requested target groups are present and disabled groups absent
python - \
    "${BASE_CKPT}" "${UNMERGED}" "${LORA_RANK}" \
    "${LORA_TARGET_ATTN}" "${LORA_TARGET_FFN}" \
    "${LORA_TARGET_ENC_VLM}" "${LORA_TARGET_ENC_WM}" "${LORA_TARGET_OUTPUT}" <<'PY'
import re,sys,torch
bp,cp,rank_s,attn_s,ffn_s,vlm_s,wm_s,out_s=sys.argv[1:9]
rank=int(rank_s)
def as_bool(x): return str(x).lower()=="true"
attn,ffn,vlm,wm,out=map(as_bool,(attn_s,ffn_s,vlm_s,wm_s,out_s))
def load(p):
    try:return torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try:return torch.load(p,map_location="cpu",weights_only=True)
        except TypeError:return torch.load(p,map_location="cpu")
base,cur=load(bp),load(cp)
bkeys={k for k,v in base.items() if torch.is_tensor(v)}
ckeys={k for k,v in cur.items() if torch.is_tensor(v)}
missing=sorted(bkeys-ckeys)
if missing:
    raise RuntimeError(f"Base keys missing: {missing[:20]}")
changed=[]
for k in sorted(bkeys):
    a,b=base[k],cur[k]
    if tuple(a.shape)!=tuple(b.shape) or a.dtype!=b.dtype or not torch.equal(a,b):
        changed.append(k)
if changed:
    raise RuntimeError(
        f"FROZEN BASE TENSORS CHANGED: count={len(changed)}, examples={changed[:30]}"
    )
extra=sorted(ckeys-bkeys)
bad=[k for k in extra if not (k.endswith(".lora_A") or k.endswith(".lora_B"))]
if bad:
    raise RuntimeError(f"Unexpected extra tensors: {bad[:30]}")

canon_A=sorted(k for k in extra if k.startswith("policy_backend.flow.") and k.endswith(".lora_A"))
canon_B=sorted(k for k in extra if k.startswith("policy_backend.flow.") and k.endswith(".lora_B"))
alias_A=sorted(k for k in extra if k.startswith("policy_action_head.") and k.endswith(".lora_A"))
alias_B=sorted(k for k in extra if k.startswith("policy_action_head.") and k.endswith(".lora_B"))
if not canon_A or len(canon_A)!=len(canon_B):
    raise RuntimeError(f"Bad canonical LoRA A/B count: {len(canon_A)}/{len(canon_B)}")
if len(alias_A)!=len(canon_A) or len(alias_B)!=len(canon_B):
    raise RuntimeError(
        f"Alias LoRA count mismatch: canonical={len(canon_A)}/{len(canon_B)}, "
        f"alias={len(alias_A)}/{len(alias_B)}"
    )

def group(prefix):
    s=prefix[len("policy_backend.flow."):]
    if re.match(r"^DiT\.transformer_blocks\.\d+\.attn1\.",s): return "attn"
    if re.match(r"^DiT\.transformer_blocks\.\d+\.ff\.",s): return "ffn"
    if s.startswith("enc_vlm."): return "enc_vlm"
    if s.startswith("enc_wm."): return "enc_wm"
    if s in ("DiT.proj_out_1","DiT.proj_out_2"): return "output"
    return "unexpected"

enabled={"attn":attn,"ffn":ffn,"enc_vlm":vlm,"enc_wm":wm,"output":out}
groups={g:0 for g in enabled}
unexpected=[]
for akey in canon_A:
    prefix=akey[:-len(".lora_A")]
    g=group(prefix)
    if g=="unexpected" or not enabled.get(g,False):
        unexpected.append(prefix)
    else:
        groups[g]+=1
if unexpected:
    raise RuntimeError(f"LoRA injected into unexpected targets: {unexpected[:30]}")
for g,on in enabled.items():
    if on and groups[g]==0:
        raise RuntimeError(f"Requested target group {g} but no LoRA module was found")
    if not on and groups[g]!=0:
        raise RuntimeError(f"Disabled target group {g} unexpectedly has {groups[g]} modules")

if any(int(cur[k].shape[0])!=rank for k in canon_A):
    raise RuntimeError("LoRA rank mismatch")
params=sum(cur[k].numel() for k in canon_A+canon_B)
B_abs=sum(float(cur[k].float().abs().sum().item()) for k in canon_B)
if B_abs<=0:
    raise RuntimeError("All canonical LoRA B tensors remain zero; adapter did not train")
print(
    "[broad-lora-check] "
    f"base_checked={len(bkeys)}, base_changed=0, "
    f"canonical_modules={len(canon_A)}, canonical_lora_params={params:,}, "
    f"group_modules={groups}, B_abs_sum={B_abs:.6g}, exact_base=True"
)
PY

# Existing generic merger handles every discovered LoRA A/B pair.
python "${MERGE_SCRIPT}" \
    "${UNMERGED}" \
    --output "${MERGED}" \
    --adapter-output "${ADAPTER}" \
    --alpha "${LORA_ALPHA}"

# HARD merged audit: only weights associated with actual LoRA tensors may differ.
python - "${BASE_CKPT}" "${UNMERGED}" "${MERGED}" <<'PY'
import sys,torch
bp,up,mp=sys.argv[1:4]
def load(p):
    try:return torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try:return torch.load(p,map_location="cpu",weights_only=True)
        except TypeError:return torch.load(p,map_location="cpu")
base,unmerged,merged=load(bp),load(up),load(mp)
bk={k for k,v in base.items() if torch.is_tensor(v)}
mk={k for k,v in merged.items() if torch.is_tensor(v)}
if bk!=mk:
    raise RuntimeError(
        f"Merged key mismatch missing={sorted(bk-mk)[:10]} extra={sorted(mk-bk)[:10]}"
    )
allowed={
    k[:-len(".lora_A")]+".weight"
    for k in unmerged
    if k.endswith(".lora_A")
}
changed=[]; bad=[]
for k in sorted(bk):
    a,b=base[k],merged[k]
    if tuple(a.shape)!=tuple(b.shape) or a.dtype!=b.dtype or not torch.equal(a,b):
        changed.append(k)
        if k not in allowed:
            bad.append(k)
if bad:
    raise RuntimeError(f"Merged checkpoint changed non-target Base tensors: {bad[:30]}")
if not changed:
    raise RuntimeError("Merged checkpoint has no changed LoRA target weights")
print(
    f"[broad-lora-merged-check] changed_target_weights={len(changed)}, "
    "changed_non_target=0, standard_keys=True"
)
PY

# T9 rollout evaluation.
unset CUDA_VISIBLE_DEVICES || true
unset NUM_PROCESSES || true
export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

EVAL_MASTER="${OUTPUT_ROOT}/t9_${PIPELINE_TIMESTAMP}"
ALIAS="broad_lora_${FLOW_LORA_TAG}_T9"
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
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${MERGED}"

EVAL_DIR=$(find "${EVAL_MASTER}/${ALIAS}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
[ -n "${EVAL_DIR}" ] || { echo "[ERROR] T9 evaluation output not found"; exit 1; }
SUITE_DIR="${EVAL_DIR}/suites/libero_goal"
python "${SUMMARY_SCRIPT}" --run-dir "${SUITE_DIR}" --task-ids 9 --expected-trials "${NUM_TRIALS}"

RESULT="${SUITE_DIR}/per_task_summary.csv"
[ -f "${RESULT}" ] || { echo "[ERROR] Missing ${RESULT}"; exit 1; }
echo "${RESULT}" > "${EXPERIMENT_ROOT}/LATEST_RESULT.txt"
echo "${RUN}" > "${EXPERIMENT_ROOT}/LATEST_RUN.txt"

echo "=========================================================="
echo " T9 Broad Flow-LoRA complete: ${TARGET_TAG}"
echo "=========================================================="
echo "Run      : ${RUN}"
echo "Adapter  : ${ADAPTER}"
echo "Merged   : ${MERGED}"
echo "Result   : ${RESULT}"
echo "Log      : ${MASTER_LOG}"
echo "=========================================================="
