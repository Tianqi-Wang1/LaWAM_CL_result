#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Strict Freeze-VLM-Interface + Independent Oracle Flow Heads
# ALIAS-AWARE REPRODUCTION v2
#
# Purpose:
#   Re-run the old Strict FreezeVLM-OH diagnostic to resolve the historical
#   contradiction, while correcting the Flow checkpoint alias issue.
#
# EXACT training protocol:
#   CL1: (S_Base, F_Base) --T6--> (S1, F6)
#   CL2: (S1,     F_Base) --T7--> (S2, F7)
#   CL3: (S2,     F_Base) --T8--> (S3, F8)
#   CL4: (S3,     F_Base) --T9--> (S4, F9)
#
# Thus F6/F7/F8/F9 are independently adapted from the SAME F_Base.
#
# Strict frozen interface:
#   policy_backend.vlm.*
#   policy_backend.act_query
#   policy_backend.flow_action_query
#
# Continually trainable shared state:
#   policy_backend.vlm_to_lam.*    (QFormer/VLMToLAM)
#   policy_backend.lam.decoder.*   (LaWM decoder)
#
# Stage-specific trainable head:
#   Flow action head
#
# CRITICAL FIX vs old 2026-08-17 script:
#   Every Flow reset/routing operation replaces BOTH:
#     policy_backend.flow.*
#     policy_action_head.*
#
# We deliberately keep the old formal hyperparameters:
#   4 GPUs, BS/GPU=32, grad accumulation=2, global batch=256,
#   2000 steps/stage, warmup=120.
#
# Execution order:
#   Goal train CL1-CL4
#   Object train CL1-CL4
#   hard-audit both chains
#   Goal oracle evaluation
#   Object oracle evaluation
#
# Modes:
#   MODE=all   (default)
#   MODE=train
#   MODE=eval
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"

COMPOSE_SCRIPT="${ROOT}/scripts/compose_lawam_flow_run_alias_aware_v2.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
CL_METRICS_SCRIPT="${ROOT}/scripts/compute_libero_cl_metrics.py"

for f in "${COMPOSE_SCRIPT}" "${SUMMARY_SCRIPT}" "${CL_METRICS_SCRIPT}"; do
    [ -f "${f}" ] || { echo "[ERROR] missing ${f}"; exit 1; }
done

MODE="${MODE:-all}"
case "${MODE}" in all|train|eval) ;; *) echo "[ERROR] MODE=${MODE}"; exit 1 ;; esac

TRAIN_GOAL="${TRAIN_GOAL:-true}"
TRAIN_OBJECT="${TRAIN_OBJECT:-true}"
GOAL_START_STAGE="${GOAL_START_STAGE:-1}"
OBJECT_START_STAGE="${OBJECT_START_STAGE:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
POLICY_GPU="${POLICY_GPU:-4}"
EVAL_GPU="${EVAL_GPU:-5}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
ORIGINAL_LR="${ORIGINAL_LR:-0.0001}"

NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
KEEP_COMPOSED_RUNS="${KEEP_COMPOSED_RUNS:-false}"
COMPUTE_CL_METRICS="${COMPUTE_CL_METRICS:-true}"

IFS=',' read -ra GPU_ARR <<< "${TRAIN_GPUS}"
[ "${#GPU_ARR[@]}" -eq 4 ] || {
    echo "[ERROR] exact reproduction expects four training GPUs."
    exit 1
}
TRAIN_NUM_PROCESSES=4
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * TRAIN_NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))

GOAL_BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
OBJECT_BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"

GOAL_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/strict_freeze_vlm_oracle_flow_aliasaware_v2"
OBJECT_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/strict_freeze_vlm_oracle_flow_aliasaware_v2"

GOAL_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/strict_freeze_vlm_oracle_flow_aliasaware_v2"
OBJECT_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_object/strict_freeze_vlm_oracle_flow_aliasaware_v2"

LOG_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/strict_freeze_vlm_oracle_flow_aliasaware_v2_logs"

mkdir -p \
    "${GOAL_RUN_ROOT}" "${OBJECT_RUN_ROOT}" \
    "${GOAL_OUTPUT_ROOT}" "${OBJECT_OUTPUT_ROOT}" \
    "${LOG_ROOT}"

TS="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="${LOG_ROOT}/strict_freeze_vlm_oh_aliasaware_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

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
    local root="$1"
    local pattern="$2"
    find "${root}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}

verify_run() {
    local label="$1"
    local run="$2"
    [ -n "${run}" ] || { echo "[ERROR] ${label}: run not found"; exit 1; }
    for f in \
        "${run}/final_model/pytorch_model.pt" \
        "${run}/dataset_statistics.json" \
        "${run}/config.yaml"
    do
        [ -f "${f}" ] || { echo "[ERROR] ${label}: missing ${f}"; exit 1; }
    done
    echo "[OK] ${label}: ${run}"
}

verify_stats() {
    local ref="$1"
    local cur="$2"
    python - "${ref}" "${cur}" <<'PY'
import json, sys
a = json.load(open(sys.argv[1], "r", encoding="utf-8"))
b = json.load(open(sys.argv[2], "r", encoding="utf-8"))
for tag in a:
    if tag not in b:
        raise RuntimeError(f"missing tag {tag}")
    for sec in ("action", "state"):
        if a[tag][sec] != b[tag][sec]:
            raise RuntimeError(f"statistics changed: {tag}/{sec}")
print("[OK] action/state normalization identical to Base.")
PY
}

verify_freeze_support() {
    python - <<'PY'
from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    LatentWorldPolicyFreezeConfig,
)
need = {
    "freeze_vlm_all",
    "freeze_act_query",
    "freeze_flow_action_query",
    "train_flow_only",
}
have = set(LatentWorldPolicyFreezeConfig.__dataclass_fields__)
missing = sorted(need - have)
if missing:
    raise RuntimeError(f"freeze_policy missing fields: {missing}")
print("[OK] current freeze_policy supports strict VLM-interface isolation.")
PY
}
verify_freeze_support

if [ -n "${GOAL_BASE_RUN:-}" ]; then
    GOAL_BASE="${GOAL_BASE_RUN}"
else
    GOAL_BASE=$(find_run "${GOAL_BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Goal Base" "${GOAL_BASE}"

if [ -n "${OBJECT_BASE_RUN:-}" ]; then
    OBJECT_BASE="${OBJECT_BASE_RUN}"
else
    OBJECT_BASE=$(find_run "${OBJECT_BASE_ROOT}" '*+base_t0_5_10k_4gpu_bs32_ga2')
fi
verify_run "Object Base" "${OBJECT_BASE}"

compose_run() {
    local shared="$1"
    local donor="$2"
    local out="$3"

    python "${COMPOSE_SCRIPT}" \
        --shared-run "${shared}" \
        --flow-run "${donor}" \
        --output-run "${out}" \
        --overwrite \
        --verify-output
}

remove_temp() {
    local d="$1"
    if [ "${KEEP_COMPOSED_RUNS}" = "true" ]; then
        echo "[INFO] keeping composed run: ${d}"
    else
        rm -rf "${d}"
    fi
}

verify_config() {
    local config="$1"
    local label="$2"
    python - "${config}" "${label}" "${ORIGINAL_LR}" <<'PY'
import math, sys
from omegaconf import OmegaConf

path, label, lr_s = sys.argv[1:4]
lr = float(lr_s)
cfg = OmegaConf.load(path)
fr = cfg.trainer.freeze

expected = {
    "freeze_vlm_all": True,
    "freeze_act_query": True,
    "freeze_flow_action_query": True,
    "train_flow_only": False,
    "unfreeze_lam_decoder": True,
}
bad = []
for k,v in expected.items():
    a = fr.get(k, None)
    if a != v:
        bad.append((k,a,v))

for group in ("vlm","action_model","world_model"):
    a = float(cfg.trainer.learning_rate[group].lr)
    if not math.isclose(a, lr, rel_tol=0.0, abs_tol=1e-12):
        bad.append((f"lr.{group}",a,lr))

if bad:
    raise RuntimeError(f"{label}: config mismatch: {bad}")

print(f"[OK] {label}: strict FreezeVLM-OH config verified.")
PY
}

# Hard checkpoint audit:
# 1) strict interface exactly Base
# 2) BOTH Flow namespaces exist and are internally identical
# 3) report QFormer/LaWM/Flow changes vs Base
audit_stage() {
    local base_ckpt="$1"
    local stage_ckpt="$2"
    local label="$3"

    python - "${base_ckpt}" "${stage_ckpt}" "${label}" <<'PY'
import gc, sys, torch

bp, sp, label = sys.argv[1:4]
CP="policy_backend.flow."
AP="policy_action_head."

def load(path):
    kw=dict(map_location="cpu")
    try: x=torch.load(path, weights_only=True, mmap=True, **kw)
    except TypeError:
        try: x=torch.load(path, weights_only=True, **kw)
        except TypeError: x=torch.load(path, **kw)
    if isinstance(x,dict):
        for w in ("state_dict","model","module"):
            n=x.get(w,None)
            if isinstance(n,dict) and n and any(torch.is_tensor(v) for v in n.values()):
                x=n; break
    return x

base, cur = load(bp), load(sp)

def group_keys(state, prefix, exact=False):
    if exact:
        return {
            k for k,v in state.items()
            if (k == prefix or k.startswith(prefix+".")) and torch.is_tensor(v)
        }
    return {k for k,v in state.items() if k.startswith(prefix) and torch.is_tensor(v)}

def compare(prefix, exact=False):
    a=group_keys(base,prefix,exact)
    b=group_keys(cur,prefix,exact)
    miss=a-b; extra=b-a
    ch=[]
    for k in sorted(a & b):
        x,y=base[k],cur[k]
        if tuple(x.shape)!=tuple(y.shape) or x.dtype!=y.dtype or not torch.equal(x,y):
            ch.append(k)
    print(f"  {prefix:<38s} base={len(a):4d} cur={len(b):4d} changed={len(ch):4d}")
    return a,b,miss,extra,ch

print(f"\n[strict-freeze-vlm-oh-audit] {label}")
for p, exact in (
    ("policy_backend.vlm.",False),
    ("policy_backend.act_query",True),
    ("policy_backend.flow_action_query",True),
):
    a,b,m,e,ch=compare(p,exact)
    if not a or not b or m or e or ch:
        raise RuntimeError(
            f"{label}: frozen interface changed at {p}; "
            f"changed={ch[:8]}, missing={list(m)[:4]}, extra={list(e)[:4]}"
        )

compare("policy_backend.vlm_to_lam.",False)
compare("policy_backend.lam.decoder.",False)
_,_,_,_,canon_changed=compare(CP,False)
_,_,_,_,alias_changed=compare(AP,False)

canon={k[len(CP):]:k for k,v in cur.items() if k.startswith(CP) and torch.is_tensor(v)}
alias={k[len(AP):]:k for k,v in cur.items() if k.startswith(AP) and torch.is_tensor(v)}
if not canon or not alias or set(canon)!=set(alias):
    raise RuntimeError(
        f"{label}: canonical/alias Flow namespace mismatch; "
        f"canonical={len(canon)} alias={len(alias)}"
    )
bad=[]
for s in sorted(canon):
    if not torch.equal(cur[canon[s]], cur[alias[s]]):
        bad.append(s)
if bad:
    raise RuntimeError(
        f"{label}: canonical/alias Flow differ internally: {bad[:12]}"
    )

print(
    f"[OK] {label}: strict interface exact; "
    f"canonical_flow={len(canon)}, alias_flow={len(alias)}, "
    f"alias_consistent=True"
)
del base,cur
gc.collect()
PY
}

train_suite() {
    local suite="$1"
    local base="$2"
    local root="$3"
    local start_stage="$4"

    local base_ckpt="${base}/final_model/pytorch_model.pt"
    local base_stats="${base}/dataset_statistics.json"

    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES="${TRAIN_NUM_PROCESSES}"

    local tmp_root="${root}/_tmp_train_init_${TS}_$$"
    mkdir -p "${tmp_root}"

    local prev_shared
    if [ "${start_stage}" -eq 1 ]; then
        prev_shared="${base}"
    else
        local ps=$((start_stage-1))
        local pt=$((ps+5))
        prev_shared=$(find_run "${root}" "*+cl${ps}_t${pt}_2k_4gpu_bs32_ga2_strict_freeze_vlm_oh_aliasaware_v2")
        verify_run "${suite} existing CL${ps}" "${prev_shared}"
    fi

    for stage in $(seq "${start_stage}" 4); do
        local task=$((stage+5))
        local run_id="cl${stage}_t${task}_2k_4gpu_bs32_ga2_strict_freeze_vlm_oh_aliasaware_v2"
        local init_run
        local init_ckpt

        if [ "${stage}" -eq 1 ]; then
            init_run="${base}"
            init_ckpt="${base_ckpt}"
        else
            init_run="${tmp_root}/${suite}_init_cl${stage}_shared_prev_plus_BaseFlow"
            compose_run "${prev_shared}" "${base}" "${init_run}"
            init_ckpt="${init_run}/final_model/pytorch_model.pt"
        fi

        echo
        echo "=========================================================="
        echo " ${suite}: Strict FreezeVLM-OH training CL${stage}/T${task}"
        echo "=========================================================="
        echo "shared source       : ${prev_shared}"
        echo "Flow donor/init     : Base"
        echo "init run            : ${init_run}"
        echo "run id              : ${run_id}"
        echo "Flow namespaces     : canonical + alias (both reset)"
        echo "=========================================================="

        bash train_lawam.sh \
            --run_root_dir="${root}" \
            --run_id="${run_id}" \
            --datasets.vla_data.cl_suite="${suite}" \
            "--datasets.vla_data.cl_task_ids=[${task}]" \
            --datasets.vla_data.use_task_filtered_statistics=false \
            --trainer.use_pretrained_dataset_statistics=true \
            --trainer.pretrained_checkpoint="${init_ckpt}" \
            --trainer.load_pretrained_policy_flow=true \
            --trainer.freeze.freeze_vision_backbone=true \
            --trainer.freeze.freeze_llm_backbone=true \
            --trainer.freeze.freeze_last_llm_layer=true \
            --trainer.freeze.freeze_embedding=true \
            --trainer.freeze.unfreeze_vision_merger=false \
            --trainer.freeze.keep_llm_first_n_layers=16 \
            --trainer.freeze.unfreeze_llm_last_n_layers=-1 \
            --trainer.freeze.freeze_vlm_all=true \
            --trainer.freeze.freeze_act_query=true \
            --trainer.freeze.freeze_flow_action_query=true \
            --trainer.freeze.train_flow_only=false \
            --trainer.freeze.unfreeze_lam_decoder=true \
            --trainer.learning_rate.vlm.lr="${ORIGINAL_LR}" \
            --trainer.learning_rate.action_model.lr="${ORIGINAL_LR}" \
            --trainer.learning_rate.world_model.lr="${ORIGINAL_LR}" \
            --datasets.vla_data.per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
            --datasets.vla_data.num_workers="${NUM_WORKERS}" \
            --datasets.vla_data.val_num_workers="${VAL_NUM_WORKERS}" \
            --datasets.vla_data.persistent_workers=true \
            --trainer.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
            --trainer.max_train_steps="${MAX_TRAIN_STEPS}" \
            --trainer.num_warmup_steps="${NUM_WARMUP_STEPS}" \
            --trainer.logging_frequency="${LOGGING_FREQUENCY}" \
            --trainer.eval_interval="${TRAIN_EVAL_INTERVAL}" \
            --trainer.eval_batches="${TRAIN_EVAL_BATCHES}" \
            --trainer.save_interval="${SAVE_INTERVAL}"

        local cur
        cur=$(find_run "${root}" "*+${run_id}")
        verify_run "${suite} CL${stage}" "${cur}"
        verify_stats "${base_stats}" "${cur}/dataset_statistics.json"
        verify_config "${cur}/config.yaml" "${suite} CL${stage}"
        audit_stage "${base_ckpt}" "${cur}/final_model/pytorch_model.pt" "${suite} CL${stage}"

        echo "[OK] ${suite} CL${stage}: independent head F${stage} saved."

        if [ "${stage}" -gt 1 ]; then
            remove_temp "${init_run}"
        fi

        # Only the SHARED state is conceptually inherited.
        # Before the next stage, BOTH Flow namespaces are reset to F_Base.
        prev_shared="${cur}"
    done

    rm -rf "${tmp_root}" || true
    unset CUDA_VISIBLE_DEVICES || true
}

resolve_chain() {
    local suite="$1"
    local base="$2"
    local root="$3"

    C1=$(find_run "${root}" '*+cl1_t6_2k_4gpu_bs32_ga2_strict_freeze_vlm_oh_aliasaware_v2')
    C2=$(find_run "${root}" '*+cl2_t7_2k_4gpu_bs32_ga2_strict_freeze_vlm_oh_aliasaware_v2')
    C3=$(find_run "${root}" '*+cl3_t8_2k_4gpu_bs32_ga2_strict_freeze_vlm_oh_aliasaware_v2')
    C4=$(find_run "${root}" '*+cl4_t9_2k_4gpu_bs32_ga2_strict_freeze_vlm_oh_aliasaware_v2')

    verify_run "${suite} CL1" "${C1}"
    verify_run "${suite} CL2" "${C2}"
    verify_run "${suite} CL3" "${C3}"
    verify_run "${suite} CL4" "${C4}"

    local bc="${base}/final_model/pytorch_model.pt"
    local bs="${base}/dataset_statistics.json"
    for item in "CL1:${C1}" "CL2:${C2}" "CL3:${C3}" "CL4:${C4}"; do
        local label="${item%%:*}"
        local run="${item#*:}"
        verify_stats "${bs}" "${run}/dataset_statistics.json"
        verify_config "${run}/config.yaml" "${suite} ${label}"
        audit_stage "${bc}" "${run}/final_model/pytorch_model.pt" "${suite} ${label}"
    done
}

run_eval_group() {
    local suite="$1"
    local master="$2"
    local manifest="$3"
    local stage="$4"
    local head="$5"
    local shared="$6"
    local donor="$7"
    local tasks="$8"
    local tmp_root="$9"

    local eval_run
    local composed=""

    if [ "${shared}" = "${donor}" ]; then
        eval_run="${shared}"
    else
        composed="${tmp_root}/${suite}_${stage}_${head}"
        compose_run "${shared}" "${donor}" "${composed}"
        eval_run="${composed}"
    fi

    local alias="${suite}_${stage}_head_${head}_aliasaware"
    local ckpt="${eval_run}/final_model/pytorch_model.pt"

    echo
    echo "----------------------------------------------------------"
    echo "${suite} ${stage} | oracle head=${head} | tasks=${tasks}"
    echo "shared=${shared}"
    echo "donor =${donor}"
    echo "----------------------------------------------------------"

    SUITES="${suite}" \
    TASK_IDS="${tasks}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${EVAL_WORKERS}" \
    GPU_IDS="${POLICY_GPU}" \
    EVAL_GPU_IDS="${EVAL_GPU}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${master}" \
    LIBERO_CKPT_ALIAS="${alias}" \
    bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "${ckpt}"

    local d
    d=$(find "${master}/${alias}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    local suite_dir="${d}/suites/${suite}"
    [ -f "${suite_dir}/episodes.jsonl" ] || {
        echo "[ERROR] missing episodes.jsonl: ${suite_dir}"
        exit 1
    }

    read -r -a ta <<< "${tasks}"
    python "${SUMMARY_SCRIPT}" \
        --run-dir "${suite_dir}" \
        --task-ids "${ta[@]}" \
        --expected-trials "${NUM_TRIALS}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${stage}" "${head}" "${shared}" "${donor}" "${suite_dir}" "${tasks}" \
        >> "${manifest}"

    if [ -n "${composed}" ]; then
        remove_temp "${composed}"
    fi
}

build_matrix() {
    local manifest="$1"
    local master="$2"
    local suite="$3"

    python - "${manifest}" "${master}" "${suite}" "${NUM_TRIALS}" <<'PY'
import csv, json, sys
from pathlib import Path

manifest=Path(sys.argv[1])
master=Path(sys.argv[2])
suite=sys.argv[3]
n=int(sys.argv[4])

with manifest.open("r",encoding="utf-8") as f:
    rows=list(csv.DictReader(f,delimiter="\t"))

order=["Base","CL1","CL2","CL3","CL4"]
res={s:{} for s in order}
long=[]

for item in rows:
    p=Path(item["eval_run_dir"])/"per_task_summary.json"
    data=json.load(open(p,"r",encoding="utf-8"))
    for r in data:
        tid=int(r["task_id"])
        trials=int(r["trials"])
        if trials != n:
            raise RuntimeError(f"{suite}/{item['stage']}/T{tid}: {trials} != {n}")
        if tid in res[item["stage"]]:
            raise RuntimeError(f"duplicate {item['stage']}/T{tid}")
        sr=float(r["success_rate"])
        res[item["stage"]][tid]=sr
        long.append({
            "suite":suite,
            "stage":item["stage"],
            "oracle_head":item["head"],
            "task_id":tid,
            "successes":int(r["successes"]),
            "trials":trials,
            "success_rate":sr,
            "eval_run_dir":item["eval_run_dir"],
        })

expected={
    "Base":set(range(6)),
    "CL1":set(range(7)),
    "CL2":set(range(8)),
    "CL3":set(range(9)),
    "CL4":set(range(10)),
}
for s in order:
    got=set(res[s])
    if got != expected[s]:
        raise RuntimeError(
            f"{suite}/{s}: missing={sorted(expected[s]-got)} "
            f"extra={sorted(got-expected[s])}"
        )

lp=master/"oracle_flow_per_task_long.csv"
with lp.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(long[0].keys()))
    w.writeheader(); w.writerows(long)

mp=master/"cl_performance_matrix.csv"
fields=["stage"]+[f"task_{i}" for i in range(10)]+["mean_seen_task_sr"]
with mp.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for s in order:
        row={"stage":s}
        for i in range(10):
            row[f"task_{i}"]=f"{res[s][i]:.4f}" if i in res[s] else ""
        row["mean_seen_task_sr"]=f"{sum(res[s].values())/len(res[s]):.4f}"
        w.writerow(row)

print(f"\nAlias-aware Strict FreezeVLM-OH matrix: {suite}")
print("Stage | "+" | ".join(f"T{i}" for i in range(10))+" | Mean")
print("-"*108)
for s in order:
    vals=[f"{res[s][i]:.2f}" if i in res[s] else "-" for i in range(10)]
    print(f"{s:>4s} | "+" | ".join(f"{v:>4s}" for v in vals)
          +f" | {sum(res[s].values())/len(res[s]):.4f}")
print("[OK]",mp)
PY
}

evaluate_suite() {
    local suite="$1"
    local base="$2"
    local root="$3"
    local output_root="$4"

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true

    export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
    export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
    export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

    resolve_chain "${suite}" "${base}" "${root}"
    local -a runs=("${C1}" "${C2}" "${C3}" "${C4}")

    local ets="$(date +"%Y%m%d_%H%M%S")"
    local master="${output_root}/oracle_aliasaware_${ets}"
    local tmp="${master}/_composed"
    mkdir -p "${master}" "${tmp}"

    local manifest="${master}/eval_manifest.tsv"
    printf "stage\thead\tshared_run\tflow_donor_run\teval_run_dir\ttask_ids\n" > "${manifest}"

    run_eval_group \
        "${suite}" "${master}" "${manifest}" \
        "Base" "Base" "${base}" "${base}" "0 1 2 3 4 5" "${tmp}"

    for idx in 0 1 2 3; do
        local stage=$((idx+1))
        local shared="${runs[$idx]}"

        run_eval_group \
            "${suite}" "${master}" "${manifest}" \
            "CL${stage}" "Base" "${shared}" "${base}" "0 1 2 3 4 5" "${tmp}"

        for h in $(seq 1 "${stage}"); do
            local donor="${runs[$((h-1))]}"
            local task=$((h+5))
            run_eval_group \
                "${suite}" "${master}" "${manifest}" \
                "CL${stage}" "CL${h}" "${shared}" "${donor}" "${task}" "${tmp}"
        done
    done

    build_matrix "${manifest}" "${master}" "${suite}"

    if [ "${COMPUTE_CL_METRICS}" = "true" ]; then
        python "${CL_METRICS_SCRIPT}" \
            "${master}/cl_performance_matrix.csv" \
            --names "strict_freeze_vlm_oh_aliasaware_${suite}" \
            --base-tasks 0 1 2 3 4 5 \
            --cl-tasks 6 7 8 9 \
            --output-dir "${master}/cl_metrics"
    fi

    cat > "${master}/ORACLE_PROTOCOL.txt" <<EOF
Strict Freeze-VLM-Interface Oracle Flow-Head Isolation, alias-aware v2.

Frozen to Base:
  policy_backend.vlm.*
  policy_backend.act_query
  policy_backend.flow_action_query

Continually trained:
  policy_backend.vlm_to_lam.*
  policy_backend.lam.decoder.*

Independent heads:
  F6/F7/F8/F9 are each initialized from F_Base.

Oracle routing:
  T0-T5 -> F_Base
  T6    -> F6
  T7    -> F7
  T8    -> F8
  T9    -> F9

CRITICAL IMPLEMENTATION:
Both Flow checkpoint namespaces are reset/routed together:
  policy_backend.flow.*
  policy_action_head.*
EOF

    rm -rf "${tmp}" || true
    echo "[OK] ${suite} evaluation: ${master}"
}

echo
echo "=========================================================="
echo " Strict FreezeVLM-OH Alias-Aware Reproduction v2"
echo "=========================================================="
echo "MODE          : ${MODE}"
echo "TRAIN_GPUS    : ${TRAIN_GPUS}"
echo "batch/GPU     : ${PER_DEVICE_BATCH_SIZE}"
echo "grad accum    : ${GRADIENT_ACCUMULATION_STEPS}"
echo "global batch  : ${GLOBAL_BATCH}"
echo "steps/stage   : ${MAX_TRAIN_STEPS}"
echo "policy GPU    : ${POLICY_GPU}"
echo "sim GPU       : ${EVAL_GPU}"
echo "trials/task   : ${NUM_TRIALS}"
echo "Flow reset    : policy_backend.flow.* + policy_action_head.*"
echo "Master log    : ${MASTER_LOG}"
echo "=========================================================="

if [ "${MODE}" = "all" ] || [ "${MODE}" = "train" ]; then
    if [ "${TRAIN_GOAL}" = "true" ]; then
        train_suite "libero_goal" "${GOAL_BASE}" "${GOAL_RUN_ROOT}" "${GOAL_START_STAGE}"
    fi
    if [ "${TRAIN_OBJECT}" = "true" ]; then
        train_suite "libero_object" "${OBJECT_BASE}" "${OBJECT_RUN_ROOT}" "${OBJECT_START_STAGE}"
    fi
fi

# Hard audit both complete chains before any evaluation.
if [ "${MODE}" = "all" ] || [ "${MODE}" = "eval" ]; then
    evaluate_suite "libero_goal" "${GOAL_BASE}" "${GOAL_RUN_ROOT}" "${GOAL_OUTPUT_ROOT}"
    sleep 5
    evaluate_suite "libero_object" "${OBJECT_BASE}" "${OBJECT_RUN_ROOT}" "${OBJECT_OUTPUT_ROOT}"
fi

echo
echo "=========================================================="
echo " COMPLETED REQUESTED PHASE"
echo "=========================================================="
echo "Goal ckpts : ${GOAL_RUN_ROOT}"
echo "Object     : ${OBJECT_RUN_ROOT}"
echo "Goal eval  : ${GOAL_OUTPUT_ROOT}"
echo "Object eval: ${OBJECT_OUTPUT_ROOT}"
echo "Log        : ${MASTER_LOG}"