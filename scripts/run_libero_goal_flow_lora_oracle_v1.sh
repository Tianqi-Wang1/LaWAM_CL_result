#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LaWAM LIBERO-Goal: Frozen WAM + Independent Flow-LoRA Oracle
#
# Purpose:
#   Test whether a tiny task-specific Flow-LoRA can preserve the single-task
#   plasticity of the 306M full Flow expert.
#
# Training:
#   Base -> T6 -> fresh LoRA_6
#   Base -> T7 -> fresh LoRA_7
#   Base -> T8 -> fresh LoRA_8
#   Base -> T9 -> fresh LoRA_9
#
# Every task starts independently from the SAME Base checkpoint.
# ALL Base WAM + Base Flow parameters are frozen.
# Only Flow LoRA A/B tensors are optimized.
#
# Evaluation (for this first diagnostic):
#   Task ID is intentionally used as an ORACLE to select the corresponding
#   merged LoRA checkpoint.  No task-free routing is studied here yet.
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
    [ -f "${required}" ] || { echo "[ERROR] Missing: ${required}"; exit 1; }
done

BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_ATTN="${LORA_TARGET_ATTN:-true}"
LORA_TARGET_FFN="${LORA_TARGET_FFN:-true}"

EXPERIMENT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/flow_lora_oracle/r${LORA_RANK}"
RUN_ROOT="${EXPERIMENT_ROOT}/heads"
LOG_ROOT="${EXPERIMENT_ROOT}/logs"
OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/flow_lora_oracle/r${LORA_RANK}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"
IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
NUM_TRAIN_GPUS="${#TRAIN_GPU_ARRAY[@]}"
[ "${NUM_TRAIN_GPUS}" -ge 1 ] || { echo "[ERROR] No TRAIN_GPUS provided."; exit 1; }

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
ORIGINAL_LR="${ORIGINAL_LR:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"

NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

FLOW_LORA_MODE="${FLOW_LORA_MODE:-cl}"   # cl | eval
FLOW_LORA_START_STAGE="${FLOW_LORA_START_STAGE:-1}"
FLOW_LORA_END_STAGE="${FLOW_LORA_END_STAGE:-4}"

case "${FLOW_LORA_MODE}" in cl|eval) ;; *) echo "[ERROR] FLOW_LORA_MODE must be cl or eval"; exit 1 ;; esac
if ! [[ "${FLOW_LORA_START_STAGE}" =~ ^[1-4]$ && "${FLOW_LORA_END_STAGE}" =~ ^[1-4]$ ]]; then
    echo "[ERROR] START/END stage must be in 1..4"; exit 1
fi
if [ "${FLOW_LORA_START_STAGE}" -gt "${FLOW_LORA_END_STAGE}" ]; then
    echo "[ERROR] START_STAGE > END_STAGE"; exit 1
fi

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

verify_support() {
python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import LatentWorldPolicyFreezeConfig
from starVLA.model.framework.latent_world.runtime.flow_lora import FlowLoRALinear, inject_flow_lora
fields = set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
required = {
    "train_flow_lora", "flow_lora_rank", "flow_lora_alpha",
    "flow_lora_dropout", "flow_lora_target_attention", "flow_lora_target_ffn",
}
missing = sorted(required - fields)
if missing:
    raise RuntimeError(f"Flow-LoRA freeze-policy support missing: {missing}")
print("[OK] Flow-LoRA code support detected.")
PY
}
verify_support

if [ -n "${BASE_RUN:-}" ]; then
    :
else
    BASE_RUN=$(find_run "${BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Goal Base" "${BASE_RUN}"
BASE_CKPT="${BASE_RUN}/final_model/pytorch_model.pt"
BASE_STATS="${BASE_RUN}/dataset_statistics.json"

verify_statistics() {
    local ref="$1" cur="$2"
python - "${ref}" "${cur}" <<'PY'
import json, sys
rp, cp = sys.argv[1:3]
with open(rp, "r", encoding="utf-8") as f: a=json.load(f)
with open(cp, "r", encoding="utf-8") as f: b=json.load(f)
for tag in a:
    for sec in ("action", "state"):
        if a[tag][sec] != b[tag][sec]:
            raise RuntimeError(f"Normalization statistics changed: {tag}/{sec}")
print("[OK] action/state normalization is identical to Base.")
PY
}

verify_config() {
    local path="$1" task="$2"
python - "${path}" "${task}" "${BASE_CKPT}" "${LORA_RANK}" "${LORA_ALPHA}" "${LORA_DROPOUT}" "${ORIGINAL_LR}" <<'PY'
import math, sys
from omegaconf import OmegaConf
path, task_s, base, rank_s, alpha_s, drop_s, lr_s = sys.argv[1:8]
cfg=OmegaConf.load(path); fr=cfg.trainer.freeze
expected = {
    "train_flow_only": False,
    "train_flow_lora": True,
    "flow_lora_rank": int(rank_s),
    "flow_lora_alpha": float(alpha_s),
    "flow_lora_dropout": float(drop_s),
    "flow_lora_target_attention": True,
    "flow_lora_target_ffn": True,
    "unfreeze_lam_decoder": False,
}
for k,v in expected.items():
    a=fr.get(k, None)
    if isinstance(v, float):
        if not math.isclose(float(a), v, rel_tol=0, abs_tol=1e-12): raise RuntimeError(f"{k}: {a} != {v}")
    elif a != v: raise RuntimeError(f"{k}: {a} != {v}")
if str(cfg.trainer.pretrained_checkpoint) != str(base): raise RuntimeError("Not initialized from formal Base checkpoint")
if list(cfg.datasets.vla_data.cl_task_ids) != [int(task_s)]: raise RuntimeError("Wrong task filter")
if not bool(cfg.trainer.load_pretrained_policy_flow): raise RuntimeError("load_pretrained_policy_flow must be true")
if not math.isclose(float(cfg.trainer.learning_rate.action_model.lr), float(lr_s), rel_tol=0, abs_tol=1e-12): raise RuntimeError("Wrong action_model LR")
print(f"[OK] Flow-LoRA config verified for T{task_s}.")
PY
}

# Before merging, every Base tensor must be bitwise unchanged; the only extra
# checkpoint tensors are LoRA A/B. This is the strongest freeze audit.
verify_unmerged_lora_checkpoint() {
    local cur="$1" label="$2"
python - "${BASE_CKPT}" "${cur}" "${label}" "${LORA_RANK}" <<'PY'
import sys, torch
bp, cp, label, rank_s = sys.argv[1:5]; expected_rank=int(rank_s)
def load(p):
    try: x=torch.load(p,map_location="cpu",weights_only=True,mmap=True)
    except TypeError:
        try: x=torch.load(p,map_location="cpu",weights_only=True)
        except TypeError: x=torch.load(p,map_location="cpu")
    return x
base,cur=load(bp),load(cp)
bkeys={k for k,v in base.items() if torch.is_tensor(v)}
ckeys={k for k,v in cur.items() if torch.is_tensor(v)}
missing=sorted(bkeys-ckeys)
if missing: raise RuntimeError(f"{label}: Base keys missing: {missing[:20]}")
changed=[]
for k in sorted(bkeys):
    a,b=base[k],cur[k]
    if tuple(a.shape)!=tuple(b.shape) or a.dtype!=b.dtype or not torch.equal(a,b): changed.append(k)
if changed: raise RuntimeError(f"{label}: FROZEN BASE TENSORS CHANGED: count={len(changed)} examples={changed[:30]}")
extra=sorted(ckeys-bkeys)
bad=[k for k in extra if not (k.endswith('.lora_A') or k.endswith('.lora_B'))]
if bad: raise RuntimeError(f"{label}: unexpected extra tensors: {bad[:30]}")
canon=[k for k in extra if k.startswith('policy_backend.flow.')]
alias=[k for k in extra if k.startswith('policy_action_head.')]
canon_A=[k for k in canon if k.endswith('.lora_A')]
canon_B=[k for k in canon if k.endswith('.lora_B')]
if len(canon_A)!=96 or len(canon_B)!=96:
    raise RuntimeError(f"{label}: expected 96 canonical A/B targets, got A={len(canon_A)} B={len(canon_B)}")
params=sum(cur[k].numel() for k in canon)
if expected_rank==8 and params != 2326528:
    raise RuntimeError(f"{label}: r8 canonical LoRA params={params}, expected=2326528")
if any(int(cur[k].shape[0]) != expected_rank for k in canon_A): raise RuntimeError(f"{label}: LoRA rank mismatch")
B_abs=sum(float(cur[k].float().abs().sum().item()) for k in canon_B)
if B_abs <= 0: raise RuntimeError(f"{label}: all LoRA B tensors are still zero; training likely did not update adapter")
print(f"[flow-lora-check] {label}: base_checked={len(bkeys)}, base_changed=0, canonical_lora_tensors={len(canon)}, alias_lora_tensors={len(alias)}, canonical_lora_params={params:,}, B_abs_sum={B_abs:.6g}, exact_base=True")
PY
}

verify_merged_checkpoint() {
    local merged="$1" label="$2"
python - "${BASE_CKPT}" "${merged}" "${label}" <<'PY'
import re, sys, torch
bp, mp, label=sys.argv[1:4]
def load(p):
    try: return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    except TypeError:
        try: return torch.load(p,map_location='cpu',weights_only=True)
        except TypeError: return torch.load(p,map_location='cpu')
a,b=load(bp),load(mp)
ak={k for k,v in a.items() if torch.is_tensor(v)}; bk={k for k,v in b.items() if torch.is_tensor(v)}
if ak!=bk: raise RuntimeError(f"{label}: merged key mismatch missing={sorted(ak-bk)[:10]} extra={sorted(bk-ak)[:10]}")
pat=re.compile(r'^(policy_backend\.flow|policy_action_head)\.DiT\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))\.weight$')
changed=[]; bad=[]
for k in sorted(ak):
    x,y=a[k],b[k]
    if tuple(x.shape)!=tuple(y.shape) or x.dtype!=y.dtype or not torch.equal(x,y):
        changed.append(k)
        if not pat.match(k): bad.append(k)
if bad: raise RuntimeError(f"{label}: merged checkpoint changed non-target Base tensors: {bad[:30]}")
if not changed: raise RuntimeError(f"{label}: merged checkpoint has no changed target weights")
print(f"[flow-lora-merged-check] {label}: changed_target_weights={len(changed)}, changed_non_target=0, standard_keys=True")
PY
}

run_id_for_stage() {
    local stage="$1" task=$((stage+5))
    echo "cl${stage}_t${task}_${MAX_TRAIN_STEPS}step_${NUM_TRAIN_GPUS}gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_flow_lora_r${LORA_RANK}_from_base"
}

resolve_stage_run() {
    local stage="$1" run_id
    run_id=$(run_id_for_stage "${stage}")
    find_run "${RUN_ROOT}" "*+${run_id}"
}

train_one_stage() {
    local stage="$1" task=$((stage+5)) run_id
    run_id=$(run_id_for_stage "${stage}")
    echo
    echo "=========================================================="
    echo " Flow-LoRA training: CL${stage}/T${task}"
    echo " Base: ${BASE_CKPT}"
    echo " LoRA: r=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
    echo " Targets: Attention Q/K/V/O + FFN in all 16 Flow blocks"
    echo " Trainable expected (r8): 2,326,528 params"
    echo "=========================================================="

    bash train_lawam.sh \
      "--run_root_dir=${RUN_ROOT}" \
      "--run_id=${run_id}" \
      "--datasets.vla_data.cl_suite=libero_goal" \
      "--datasets.vla_data.cl_task_ids=[${task}]" \
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
      "--trainer.freeze.unfreeze_lam_decoder=false" \
      "--trainer.learning_rate.vlm.lr=${ORIGINAL_LR}" \
      "--trainer.learning_rate.action_model.lr=${ORIGINAL_LR}" \
      "--trainer.learning_rate.world_model.lr=${ORIGINAL_LR}" \
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
}

prepare_stage_artifacts() {
    local stage="$1" task=$((stage+5)) run
    run=$(resolve_stage_run "${stage}")
    verify_run "Flow-LoRA CL${stage}/T${task}" "${run}"
    verify_statistics "${BASE_STATS}" "${run}/dataset_statistics.json"
    verify_config "${run}/config.yaml" "${task}"
    verify_unmerged_lora_checkpoint "${run}/final_model/pytorch_model.pt" "CL${stage}/T${task}"

    local merged="${run}/final_model/pytorch_model_merged.pt"
    local adapter="${run}/final_model/flow_lora_adapter.pt"
    python "${MERGE_SCRIPT}" \
        "${run}/final_model/pytorch_model.pt" \
        --output "${merged}" \
        --adapter-output "${adapter}" \
        --alpha "${LORA_ALPHA}"
    verify_merged_checkpoint "${merged}" "CL${stage}/T${task}"
    echo "${run}"
}

prepare_eval_environment() {
    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
    export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
    export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
    export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
}

evaluate_one_stage() {
    local master="$1" manifest="$2" stage="$3" run="$4"
    local task=$((stage+5)) label="CL${stage}" ckpt="${run}/final_model/pytorch_model_merged.pt"
    local alias="flow_lora_r${LORA_RANK}_${label}_T${task}"

    SUITES="libero_goal" \
    TASK_IDS="${task}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${EVAL_WORKERS}" \
    GPU_IDS="${POLICY_GPU}" \
    EVAL_GPU_IDS="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${master}" \
    LIBERO_CKPT_ALIAS="${alias}" \
    bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${ckpt}"

    local d suite_dir
    d=$(find "${master}/${alias}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    [ -n "${d}" ] || { echo "[ERROR] eval output not found: ${alias}"; exit 1; }
    suite_dir="${d}/suites/libero_goal"
    python "${SUMMARY_SCRIPT}" --run-dir "${suite_dir}" --task-ids "${task}" --expected-trials "${NUM_TRIALS}"
    printf "CL%s\t%s\t%s\t%s\t%s\n" "${stage}" "${task}" "${run}" "${suite_dir}" "${ckpt}" >> "${manifest}"
}

build_summary() {
    local manifest="$1" master="$2"
python - "${manifest}" "${master}" <<'PY'
import csv, json, sys
from pathlib import Path
manifest, master = Path(sys.argv[1]), Path(sys.argv[2])
with manifest.open('r',encoding='utf-8') as f: items=list(csv.DictReader(f,delimiter='\t'))
rows=[]
for it in items:
    p=Path(it['eval_run_dir'])/'per_task_summary.json'
    with p.open('r',encoding='utf-8') as f: r=json.load(f)
    if len(r)!=1: raise RuntimeError(f"Expected one task in {p}, got {len(r)}")
    x=r[0]
    rows.append({
      'stage':it['stage'],'task_id':int(x['task_id']),'successes':int(x['successes']),
      'trials':int(x['trials']),'success_rate':float(x['success_rate']),
      'model_run':it['model_run'],'merged_checkpoint':it['merged_checkpoint']})
out=master/'flow_lora_single_task_sr.csv'
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('\nFlow-LoRA Oracle Single-Task Results')
for r in rows: print(f"  {r['stage']}/T{r['task_id']}: {r['successes']}/{r['trials']} = {r['success_rate']:.4f}")
print(f"[OK] Saved {out}")
PY
}

PIPELINE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/flow_lora_r${LORA_RANK}_${PIPELINE_TIMESTAMP}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=========================================================="
echo " LaWAM Goal: Frozen WAM + Flow-LoRA Oracle"
echo "=========================================================="
echo "Base             : ${BASE_RUN}"
echo "Train GPUs       : ${TRAIN_GPUS} (${NUM_TRAIN_GPUS})"
echo "Global batch     : $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_GPUS))"
echo "LoRA rank/alpha  : ${LORA_RANK}/${LORA_ALPHA}"
echo "LoRA dropout     : ${LORA_DROPOUT}"
echo "Target           : Attention Q/K/V/O + FFN"
echo "Mode/stages      : ${FLOW_LORA_MODE} / ${FLOW_LORA_START_STAGE}-${FLOW_LORA_END_STAGE}"
echo "Evaluation       : oracle task-ID, ${NUM_TRIALS} trials/task"
echo "=========================================================="

if [ "${FLOW_LORA_MODE}" = "cl" ]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES="${NUM_TRAIN_GPUS}"
    for stage in $(seq "${FLOW_LORA_START_STAGE}" "${FLOW_LORA_END_STAGE}"); do
        train_one_stage "${stage}"
        prepare_stage_artifacts "${stage}"
    done
    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
fi

prepare_eval_environment
MASTER_DIR="${OUTPUT_ROOT}/single_task_${PIPELINE_TIMESTAMP}"
mkdir -p "${MASTER_DIR}"
MANIFEST="${MASTER_DIR}/manifest.tsv"
printf "stage\ttask_id\tmodel_run\teval_run_dir\tmerged_checkpoint\n" > "${MANIFEST}"

for stage in $(seq "${FLOW_LORA_START_STAGE}" "${FLOW_LORA_END_STAGE}"); do
    run=$(prepare_stage_artifacts "${stage}" | tail -n 1)
    evaluate_one_stage "${MASTER_DIR}" "${MANIFEST}" "${stage}" "${run}"
done

build_summary "${MANIFEST}" "${MASTER_DIR}"

echo "=========================================================="
echo " Flow-LoRA oracle experiment complete"
echo " Results: ${MASTER_DIR}/flow_lora_single_task_sr.csv"
echo " Log    : ${MASTER_LOG}"
echo "=========================================================="
