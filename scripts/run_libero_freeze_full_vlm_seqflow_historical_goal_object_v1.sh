#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LaWAM LIBERO Goal + Object
# FULL Freeze-VLM-Interface + Sequential Flow + Historical Flow Evaluation
#
# Frozen during CL:
#   policy_backend.vlm.*              (vision/merger/LLM/embeddings)
#   policy_backend.act_query
#   policy_backend.flow_action_query
#
# Trainable during CL:
#   policy_backend.vlm_to_lam.*       (QFormer / VLMToLAM)
#   policy_backend.lam.decoder.*      (LaWM decoder)
#   policy_backend.flow.*             (Flow action head)
#
# Flow training is SEQUENTIAL:
#   Base -> CL1 -> CL2 -> CL3 -> CL4
#
# Historical Flow evaluation at CL-k:
#   tasks 0-5 -> Base Flow
#   task 6    -> CL1 Flow
#   task 7    -> CL2 Flow
#   task 8    -> CL3 Flow
#   task 9    -> CL4 Flow
#
# Execution order:
#   Goal training -> Object training -> Goal eval -> Object eval
#
# MODE:
#   all   : train both suites, then evaluate both (default)
#   train : train both suites only
#   eval  : evaluate existing chains only
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="${ROOT:-/home/jincai_guo/tianqi/CVPR2027/LaWAM}"
cd "${ROOT}"

SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
CL_METRICS_SCRIPT="${ROOT}/scripts/compute_libero_cl_metrics.py"
FREEZE_POLICY_PY="${ROOT}/starVLA/model/framework/latent_world/runtime/freeze_policy.py"
TRAIN_YAML="${ROOT}/starVLA/config/training/train_libero.yaml"

for f in "${SUMMARY_SCRIPT}" "${CL_METRICS_SCRIPT}" "${FREEZE_POLICY_PY}" "${TRAIN_YAML}"; do
    [ -f "${f}" ] || { echo "[ERROR] Missing ${f}"; exit 1; }
done

MODE="${MODE:-all}"
case "${MODE}" in all|train|eval) ;; *) echo "[ERROR] MODE=${MODE}"; exit 1 ;; esac

TRAIN_GPUS="${TRAIN_GPUS:-1,2,6,7}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

IFS=',' read -ra TRAIN_GPU_ARRAY <<< "${TRAIN_GPUS}"
[ "${#TRAIN_GPU_ARRAY[@]}" -eq 4 ] || {
    echo "[ERROR] Exactly 4 training GPUs are required. TRAIN_GPUS=${TRAIN_GPUS}"
    exit 1
}
NUM_PROCESSES=4
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))

POLICY_GPU="${POLICY_GPU:-6}"
EVAL_GPU="${EVAL_GPU:-7}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
NUM_TRIALS="${NUM_TRIALS:-50}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-120}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
TRAIN_EVAL_INTERVAL="${TRAIN_EVAL_INTERVAL:-500}"
TRAIN_EVAL_BATCHES="${TRAIN_EVAL_BATCHES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-$((MAX_TRAIN_STEPS + 1))}"
ORIGINAL_LR="${ORIGINAL_LR:-0.0001}"

COMPUTE_CL_METRICS="${COMPUTE_CL_METRICS:-true}"
KEEP_COMPOSED_RUNS="${KEEP_COMPOSED_RUNS:-false}"
GOAL_START_STAGE="${GOAL_START_STAGE:-1}"
OBJECT_START_STAGE="${OBJECT_START_STAGE:-1}"

for s in "${GOAL_START_STAGE}" "${OBJECT_START_STAGE}"; do
    [[ "${s}" =~ ^[1-4]$ ]] || { echo "[ERROR] START_STAGE=${s}"; exit 1; }
done

BATCH_TAG="4gpu_bs${PER_DEVICE_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}"
RUN_SUFFIX="${BATCH_TAG}_freeze_full_vlm_seqflow"

GOAL_BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"
OBJECT_BASE_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft"

GOAL_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/freeze_full_vlm_world_oh"
OBJECT_RUN_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/freeze_full_vlm_world_oh"

GOAL_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_goal/freeze_full_vlm_world_oh"
OBJECT_OUTPUT_ROOT="${ROOT}/results/eval_runs/lawam_cl/libero_object/freeze_full_vlm_world_oh"
LOG_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/freeze_full_vlm_world_oh_logs"

mkdir -p "${GOAL_RUN_ROOT}" "${OBJECT_RUN_ROOT}" "${GOAL_OUTPUT_ROOT}" "${OBJECT_OUTPUT_ROOT}" "${LOG_ROOT}"

MASTER_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
MASTER_LOG="${LOG_ROOT}/goal_object_freeze_full_vlm_world_oh_${MASTER_TIMESTAMP}.log"
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
    for f in "${run}/final_model/pytorch_model.pt" "${run}/dataset_statistics.json" "${run}/config.yaml"; do
        [ -f "${f}" ] || { echo "[ERROR] ${label}: missing ${f}"; exit 1; }
    done
    echo "[OK] ${label}: ${run}"
}

verify_statistics() {
    local ref="$1"
    local cur="$2"
    python - "${ref}" "${cur}" <<'PY'
import json, sys
r, c = sys.argv[1:3]
with open(r, "r", encoding="utf-8") as f: a = json.load(f)
with open(c, "r", encoding="utf-8") as f: b = json.load(f)
for tag in a:
    if tag not in b:
        raise RuntimeError(f"Missing tag: {tag}")
    for sec in ("action", "state"):
        if a[tag][sec] != b[tag][sec]:
            raise RuntimeError(f"Normalization changed: {tag}/{sec}")
print("[OK] action/state normalization identical to Base.")
PY
}

verify_support() {
    python - "${FREEZE_POLICY_PY}" "${TRAIN_YAML}" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]).read_text(encoding="utf-8")
y = Path(sys.argv[2]).read_text(encoding="utf-8")
reqp = ["freeze_act_query", "freeze_flow_action_query", "policy_backend.act_query", "policy_backend.flow_action_query"]
reqy = ["freeze_act_query:", "freeze_flow_action_query:"]
mp = [x for x in reqp if x not in p]
my = [x for x in reqy if x not in y]
if mp or my:
    raise RuntimeError(
        "Full-VLM query-freeze support missing.\n"
        f"freeze_policy.py missing={mp}\ntrain_libero.yaml missing={my}\n"
        "Run scripts/patch_full_vlm_freeze_support.py first."
    )
print("[OK] Full-VLM query-freeze support detected.")
PY
}
verify_support

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

verify_full_vlm_config() {
    local path="$1"
    local label="$2"
    python - "${path}" "${label}" "${ORIGINAL_LR}" <<'PY'
import math, sys
from omegaconf import OmegaConf
path, label, lr_s = sys.argv[1:4]
lr = float(lr_s)
cfg = OmegaConf.load(path)
fr = cfg.trainer.freeze
expected = {
    "freeze_vision_backbone": True,
    "freeze_llm_backbone": True,
    "freeze_last_llm_layer": True,
    "freeze_embedding": True,
    "unfreeze_vision_merger": False,
    "keep_llm_first_n_layers": 16,
    "unfreeze_llm_last_n_layers": -1,
    "unfreeze_lam_decoder": True,
    "freeze_act_query": True,
    "freeze_flow_action_query": True,
}
bad = []
for k, v in expected.items():
    a = fr.get(k, None)
    if a != v: bad.append((f"trainer.freeze.{k}", a, v))
for k in ("vlm", "action_model", "world_model"):
    a = float(cfg.trainer.learning_rate[k].lr)
    if not math.isclose(a, lr, rel_tol=0.0, abs_tol=1e-12):
        bad.append((f"trainer.learning_rate.{k}.lr", a, lr))
if bad:
    raise RuntimeError(label + " config mismatch:\n" + "\n".join(f"  {k}: {a!r} != {v!r}" for k,a,v in bad))
print(f"[OK] {label}: full VLM/interface freeze config verified.")
PY
}

verify_checkpoint_against_base() {
    local base_ckpt="$1"
    local cur_ckpt="$2"
    local label="$3"
    python - "${base_ckpt}" "${cur_ckpt}" "${label}" <<'PY'
import gc, sys, torch
bp, cp, label = sys.argv[1:4]

def load(path):
    kw = dict(map_location="cpu")
    try: x = torch.load(path, weights_only=True, mmap=True, **kw)
    except TypeError:
        try: x = torch.load(path, weights_only=True, **kw)
        except TypeError: x = torch.load(path, **kw)
    if isinstance(x, dict):
        for k in ("state_dict", "model", "module"):
            n = x.get(k, None)
            if isinstance(n, dict) and n and any(torch.is_tensor(v) for v in n.values()):
                x = n
                break
    if not isinstance(x, dict): raise RuntimeError(f"Not state dict: {path}")
    return x

base, cur = load(bp), load(cp)

def keys(state, group):
    if group.endswith("."):
        return {k for k,v in state.items() if k.startswith(group) and torch.is_tensor(v)}
    return {k for k,v in state.items() if (k == group or k.startswith(group + ".")) and torch.is_tensor(v)}

def comp(group):
    a, b = keys(base, group), keys(cur, group)
    miss, extra = sorted(a-b), sorted(b-a)
    changed = []
    for k in sorted(a & b):
        x, y = base[k], cur[k]
        if tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype or not torch.equal(x, y):
            changed.append(k)
    print(f"  {group:<40s} base={len(a):4d} cur={len(b):4d} changed={len(changed):4d} missing={len(miss):3d} extra={len(extra):3d}")
    return a, b, miss, extra, changed

print(f"\n[full-vlm-freeze-check] {label}")
for g in ("policy_backend.vlm.", "policy_backend.act_query", "policy_backend.flow_action_query"):
    a,b,m,e,c = comp(g)
    if not a or not b or m or e or c:
        raise RuntimeError(f"{label}: FROZEN GROUP CHANGED: {g}; changed={c[:8]}, missing={m[:4]}, extra={e[:4]}")

for g in ("policy_backend.vlm_to_lam.", "policy_backend.lam.decoder.", "policy_backend.flow.", "policy_action_head."):
    comp(g)

print(f"[OK] {label}: VLM + merger + queries are bitwise Base-identical.")
del base, cur
gc.collect()
PY
}

train_suite() {
    local suite="$1"
    local base_run="$2"
    local run_root="$3"
    local start_stage="$4"

    local base_ckpt="${base_run}/final_model/pytorch_model.pt"
    local base_stats="${base_run}/dataset_statistics.json"

    echo
    echo "################################################################################"
    echo "# TRAIN ${suite}"
    echo "################################################################################"
    echo "Frozen    : VLM + merger + act_query + flow_action_query"
    echo "Trainable : QFormer/VLMToLAM + LaWM decoder + Flow"
    echo "GPUs      : ${TRAIN_GPUS}"
    echo "Batch     : ${PER_DEVICE_BATCH_SIZE} x 4 x ${GRADIENT_ACCUMULATION_STEPS} = ${GLOBAL_BATCH}"

    export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
    export NUM_PROCESSES="${NUM_PROCESSES}"

    local prev_run
    if [ "${start_stage}" -eq 1 ]; then
        prev_run="${base_run}"
    else
        local ps=$((start_stage - 1))
        local pt=$((ps + 5))
        prev_run=$(find_run "${run_root}" "*+cl${ps}_t${pt}_2k_${RUN_SUFFIX}")
        verify_run "${suite} existing CL${ps}" "${prev_run}"
    fi

    for stage in $(seq "${start_stage}" 4); do
        local task=$((stage + 5))
        local run_id="cl${stage}_t${task}_2k_${RUN_SUFFIX}"
        local prev_ckpt="${prev_run}/final_model/pytorch_model.pt"

        echo
        echo "=========================================================="
        echo " ${suite} CL${stage} / task ${task}"
        echo "=========================================================="
        echo "Previous: ${prev_run}"
        echo "Run ID  : ${run_id}"

        bash train_lawam.sh \
            --run_root_dir="${run_root}" \
            --run_id="${run_id}" \
            --datasets.vla_data.cl_suite="${suite}" \
            "--datasets.vla_data.cl_task_ids=[${task}]" \
            --datasets.vla_data.use_task_filtered_statistics=false \
            --trainer.use_pretrained_dataset_statistics=true \
            --trainer.pretrained_checkpoint="${prev_ckpt}" \
            --trainer.load_pretrained_policy_flow=true \
            --trainer.freeze.freeze_vision_backbone=true \
            --trainer.freeze.freeze_llm_backbone=true \
            --trainer.freeze.freeze_last_llm_layer=true \
            --trainer.freeze.freeze_embedding=true \
            --trainer.freeze.unfreeze_vision_merger=false \
            --trainer.freeze.keep_llm_first_n_layers=16 \
            --trainer.freeze.unfreeze_llm_last_n_layers=-1 \
            --trainer.freeze.unfreeze_lam_decoder=true \
            --trainer.freeze.freeze_act_query=true \
            --trainer.freeze.freeze_flow_action_query=true \
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
        cur=$(find_run "${run_root}" "*+${run_id}")
        verify_run "${suite} CL${stage}" "${cur}"
        verify_statistics "${base_stats}" "${cur}/dataset_statistics.json"
        verify_full_vlm_config "${cur}/config.yaml" "${suite} CL${stage}"
        verify_checkpoint_against_base "${base_ckpt}" "${cur}/final_model/pytorch_model.pt" "${suite} CL${stage}"
        prev_run="${cur}"
    done

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
}

compose_historical_flow() {
    local current_run="$1"
    local hist_run="$2"
    local temp_root="$3"
    local label="$4"

    local current_ckpt="${current_run}/final_model/pytorch_model.pt"
    local hist_ckpt="${hist_run}/final_model/pytorch_model.pt"
    local out_run="${temp_root}/${label}"
    local out_ckpt="${out_run}/final_model/pytorch_model.pt"

    rm -rf "${out_run}"
    mkdir -p "${out_run}/final_model"
    cp "${current_run}/config.yaml" "${out_run}/config.yaml"
    cp "${current_run}/dataset_statistics.json" "${out_run}/dataset_statistics.json"

    python - "${current_ckpt}" "${hist_ckpt}" "${out_ckpt}" "${label}" <<'PY'
import gc, sys, torch
from pathlib import Path
curp, histp, outp, label = sys.argv[1:5]

def load(path):
    kw = dict(map_location="cpu")
    try: x = torch.load(path, weights_only=True, mmap=True, **kw)
    except TypeError:
        try: x = torch.load(path, weights_only=True, **kw)
        except TypeError: x = torch.load(path, **kw)
    if not isinstance(x, dict): raise RuntimeError(f"{label}: not flat state dict")
    for k in ("state_dict", "model", "module"):
        n = x.get(k, None)
        if isinstance(n, dict) and n and any(torch.is_tensor(v) for v in n.values()):
            raise RuntimeError(f"{label}: nested state dict wrapper `{k}` detected")
    return x

def is_flow(k):
    return (
        k == "policy_backend.flow"
        or k.startswith("policy_backend.flow.")
        or k == "policy_action_head"
        or k.startswith("policy_action_head.")
    )

cur, hist = load(curp), load(histp)
cf = {k for k,v in cur.items() if is_flow(k) and torch.is_tensor(v)}
hf = {k for k,v in hist.items() if is_flow(k) and torch.is_tensor(v)}
canonical = [k for k in cf if k.startswith("policy_backend.flow.")]

if not canonical:
    raise RuntimeError(f"{label}: no canonical policy_backend.flow.* tensors")
if cf != hf:
    raise RuntimeError(
        f"{label}: flow key mismatch; missing={sorted(cf-hf)[:8]} extra={sorted(hf-cf)[:8]}"
    )

out = dict(cur)
for k in sorted(cf):
    a, b = cur[k], hist[k]
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        raise RuntimeError(f"{label}: shape/dtype mismatch at {k}")
    out[k] = b

Path(outp).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, outp)
print(f"[OK] {label}: historical Flow transplanted; flow_tensors={len(cf)}")
print(str(Path(outp).parents[1]))
del cur, hist, out
gc.collect()
PY
}

run_eval_group() {
    local suite="$1"
    local master="$2"
    local manifest="$3"
    local stage="$4"
    local routed_flow="$5"
    local run="$6"
    local tasks="$7"
    local alias="$8"

    local ckpt="${run}/final_model/pytorch_model.pt"

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
    [ -n "${d}" ] || { echo "[ERROR] eval output not found for ${alias}"; exit 1; }

    local suite_dir="${d}/suites/${suite}"
    [ -f "${suite_dir}/episodes.jsonl" ] || { echo "[ERROR] missing episodes: ${suite_dir}"; exit 1; }

    read -r -a ta <<< "${tasks}"
    python "${SUMMARY_SCRIPT}" --run-dir "${suite_dir}" --task-ids "${ta[@]}" --expected-trials "${NUM_TRIALS}"

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${stage}" "${routed_flow}" "${run}" "${suite_dir}" "${tasks}" "${alias}" >> "${manifest}"
}

build_matrix() {
    local manifest="$1"
    local master="$2"
    local suite="$3"
    python - "${manifest}" "${master}" "${suite}" "${NUM_TRIALS}" <<'PY'
import csv, json, sys
from pathlib import Path

manifest, master, suite, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
with manifest.open("r", encoding="utf-8") as f:
    items = list(csv.DictReader(f, delimiter="\t"))

order = ["Base", "CL1", "CL2", "CL3", "CL4"]
res = {s:{} for s in order}
long = []

for it in items:
    p = Path(it["eval_run_dir"]) / "per_task_summary.json"
    with p.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    for r in rows:
        tid = int(r["task_id"])
        trials = int(r["trials"])
        if trials != n: raise RuntimeError(f"{suite}/{it['stage']}/T{tid}: trials={trials} expected={n}")
        if tid in res[it["stage"]]: raise RuntimeError(f"duplicate {it['stage']}/T{tid}")
        sr = float(r["success_rate"])
        res[it["stage"]][tid] = sr
        long.append({
            "suite": suite,
            "stage": it["stage"],
            "historical_flow": it["historical_flow"],
            "task_id": tid,
            "task_description": r.get("task_description",""),
            "successes": int(r["successes"]),
            "trials": trials,
            "success_rate": sr,
            "eval_run_dir": it["eval_run_dir"],
        })

expected = {
    "Base": set(range(6)),
    "CL1": set(range(7)),
    "CL2": set(range(8)),
    "CL3": set(range(9)),
    "CL4": set(range(10)),
}
for s in order:
    got = set(res[s])
    if got != expected[s]:
        raise RuntimeError(f"{suite}/{s}: missing={sorted(expected[s]-got)} extra={sorted(got-expected[s])}")

lp = master / "historical_flow_per_task.csv"
with lp.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(long[0].keys()))
    w.writeheader(); w.writerows(long)

mp = master / "cl_performance_matrix.csv"
fields = ["stage"] + [f"task_{i}" for i in range(10)] + ["mean_seen_task_sr"]
with mp.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for s in order:
        row = {"stage": s}
        for i in range(10):
            row[f"task_{i}"] = f"{res[s][i]:.4f}" if i in res[s] else ""
        row["mean_seen_task_sr"] = f"{sum(res[s].values())/len(res[s]):.4f}"
        w.writerow(row)

print(f"\nHistorical-Flow SR matrix: {suite}")
print("Stage | " + " | ".join(f"T{i}" for i in range(10)) + " | Mean")
print("-"*108)
for s in order:
    vals = [f"{res[s][i]:.2f}" if i in res[s] else "-" for i in range(10)]
    print(f"{s:>4s} | " + " | ".join(f"{x:>4s}" for x in vals) + f" | {sum(res[s].values())/len(res[s]):.4f}")
print(f"[OK] {mp}")
print(f"[OK] {lp}")
PY
}

evaluate_suite() {
    local suite="$1"
    local base="$2"
    local c1="$3"
    local c2="$4"
    local c3="$5"
    local c4="$6"
    local output_root="$7"

    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
    export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
    export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
    export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python

    local ts="$(date +"%Y%m%d_%H%M%S")"
    local master="${output_root}/historical_flow_${ts}"
    local tmp="${master}/_composed"
    mkdir -p "${master}" "${tmp}"
    local manifest="${master}/eval_manifest.tsv"
    printf "stage\thistorical_flow\tcheckpoint_run\teval_run_dir\ttask_ids\talias\n" > "${manifest}"

    echo
    echo "################################################################################"
    echo "# EVAL ${suite}: CURRENT WORLD + HISTORICAL FLOW"
    echo "################################################################################"
    echo "Policy GPU=${POLICY_GPU} | Sim GPU=${EVAL_GPU} | Workers=${EVAL_WORKERS} | Trials=${NUM_TRIALS}"

    run_eval_group "${suite}" "${master}" "${manifest}" "Base" "BaseFlow" "${base}" "0 1 2 3 4 5" "${suite}_base_native"

    local -a runs=("${c1}" "${c2}" "${c3}" "${c4}")
    local current stage newtask comp histstage histtask histrun

    for idx in 0 1 2 3; do
        stage=$((idx + 1))
        newtask=$((stage + 5))
        current="${runs[$idx]}"

        comp=$(compose_historical_flow "${current}" "${base}" "${tmp}" "${suite}_CL${stage}_BaseFlow" | tail -n 1)
        run_eval_group "${suite}" "${master}" "${manifest}" "CL${stage}" "BaseFlow" "${comp}" "0 1 2 3 4 5" "${suite}_cl${stage}_baseflow"
        [ "${KEEP_COMPOSED_RUNS}" = "true" ] || rm -rf "${comp}"

        if [ "${stage}" -ge 2 ]; then
            for histstage in $(seq 1 $((stage - 1))); do
                histtask=$((histstage + 5))
                histrun="${runs[$((histstage - 1))]}"
                comp=$(compose_historical_flow "${current}" "${histrun}" "${tmp}" "${suite}_CL${stage}_CL${histstage}Flow" | tail -n 1)
                run_eval_group "${suite}" "${master}" "${manifest}" "CL${stage}" "CL${histstage}Flow" "${comp}" "${histtask}" "${suite}_cl${stage}_cl${histstage}flow_t${histtask}"
                [ "${KEEP_COMPOSED_RUNS}" = "true" ] || rm -rf "${comp}"
            done
        fi

        run_eval_group "${suite}" "${master}" "${manifest}" "CL${stage}" "CL${stage}Flow" "${current}" "${newtask}" "${suite}_cl${stage}_native_t${newtask}"
    done

    build_matrix "${manifest}" "${master}" "${suite}"

    if [ "${COMPUTE_CL_METRICS}" = "true" ]; then
        python "${CL_METRICS_SCRIPT}" \
            "${master}/cl_performance_matrix.csv" \
            --names "freeze_full_vlm_seqflow_historical_${suite}" \
            --base-tasks 0 1 2 3 4 5 \
            --cl-tasks 6 7 8 9 \
            --output-dir "${master}/cl_metrics"
    fi

    [ "${KEEP_COMPOSED_RUNS}" = "true" ] || rm -rf "${tmp}"
    echo "[OK] ${suite} eval complete: ${master}"
}

echo
echo "=========================================================="
echo " FULL Freeze-VLM + Sequential Flow + Historical Flow"
echo "=========================================================="
echo "MODE           : ${MODE}"
echo "TRAIN_GPUS     : ${TRAIN_GPUS}"
echo "batch/GPU      : ${PER_DEVICE_BATCH_SIZE}"
echo "grad accum     : ${GRADIENT_ACCUMULATION_STEPS}"
echo "global batch   : ${GLOBAL_BATCH}"
echo "POLICY_GPU     : ${POLICY_GPU}"
echo "EVAL_GPU       : ${EVAL_GPU}"
echo "workers        : ${EVAL_WORKERS}"
echo "trials/task    : ${NUM_TRIALS}"
echo "Execution      : Goal train -> Object train -> Goal eval -> Object eval"
echo "Master log     : ${MASTER_LOG}"
echo "=========================================================="

if [ "${MODE}" = "all" ] || [ "${MODE}" = "train" ]; then
    train_suite "libero_goal" "${GOAL_BASE}" "${GOAL_RUN_ROOT}" "${GOAL_START_STAGE}"
    train_suite "libero_object" "${OBJECT_BASE}" "${OBJECT_RUN_ROOT}" "${OBJECT_START_STAGE}"
fi

GOAL_C1=$(find_run "${GOAL_RUN_ROOT}" "*+cl1_t6_2k_${RUN_SUFFIX}")
GOAL_C2=$(find_run "${GOAL_RUN_ROOT}" "*+cl2_t7_2k_${RUN_SUFFIX}")
GOAL_C3=$(find_run "${GOAL_RUN_ROOT}" "*+cl3_t8_2k_${RUN_SUFFIX}")
GOAL_C4=$(find_run "${GOAL_RUN_ROOT}" "*+cl4_t9_2k_${RUN_SUFFIX}")
OBJECT_C1=$(find_run "${OBJECT_RUN_ROOT}" "*+cl1_t6_2k_${RUN_SUFFIX}")
OBJECT_C2=$(find_run "${OBJECT_RUN_ROOT}" "*+cl2_t7_2k_${RUN_SUFFIX}")
OBJECT_C3=$(find_run "${OBJECT_RUN_ROOT}" "*+cl3_t8_2k_${RUN_SUFFIX}")
OBJECT_C4=$(find_run "${OBJECT_RUN_ROOT}" "*+cl4_t9_2k_${RUN_SUFFIX}")

for item in \
    "Goal CL1:${GOAL_C1}" "Goal CL2:${GOAL_C2}" "Goal CL3:${GOAL_C3}" "Goal CL4:${GOAL_C4}" \
    "Object CL1:${OBJECT_C1}" "Object CL2:${OBJECT_C2}" "Object CL3:${OBJECT_C3}" "Object CL4:${OBJECT_C4}"
do
    label="${item%%:*}"; run="${item#*:}"
    verify_run "${label}" "${run}"
done

for item in "Goal CL1:${GOAL_C1}" "Goal CL2:${GOAL_C2}" "Goal CL3:${GOAL_C3}" "Goal CL4:${GOAL_C4}"; do
    label="${item%%:*}"; run="${item#*:}"
    verify_statistics "${GOAL_BASE}/dataset_statistics.json" "${run}/dataset_statistics.json"
    verify_full_vlm_config "${run}/config.yaml" "${label}"
    verify_checkpoint_against_base "${GOAL_BASE}/final_model/pytorch_model.pt" "${run}/final_model/pytorch_model.pt" "${label}"
done

for item in "Object CL1:${OBJECT_C1}" "Object CL2:${OBJECT_C2}" "Object CL3:${OBJECT_C3}" "Object CL4:${OBJECT_C4}"; do
    label="${item%%:*}"; run="${item#*:}"
    verify_statistics "${OBJECT_BASE}/dataset_statistics.json" "${run}/dataset_statistics.json"
    verify_full_vlm_config "${run}/config.yaml" "${label}"
    verify_checkpoint_against_base "${OBJECT_BASE}/final_model/pytorch_model.pt" "${run}/final_model/pytorch_model.pt" "${label}"
done

echo "[OK] Both chains passed full VLM/interface invariance checks."

if [ "${MODE}" = "all" ] || [ "${MODE}" = "eval" ]; then
    unset CUDA_VISIBLE_DEVICES || true
    unset NUM_PROCESSES || true
    sleep 5

    evaluate_suite "libero_goal" "${GOAL_BASE}" "${GOAL_C1}" "${GOAL_C2}" "${GOAL_C3}" "${GOAL_C4}" "${GOAL_OUTPUT_ROOT}"
    sleep 5
    evaluate_suite "libero_object" "${OBJECT_BASE}" "${OBJECT_C1}" "${OBJECT_C2}" "${OBJECT_C3}" "${OBJECT_C4}" "${OBJECT_OUTPUT_ROOT}"
fi

echo
echo "=========================================================="
echo " COMPLETED"
echo "=========================================================="
echo "Master log : ${MASTER_LOG}"
echo "Goal ckpts : ${GOAL_RUN_ROOT}"
echo "Object     : ${OBJECT_RUN_ROOT}"
echo "Goal eval  : ${GOAL_OUTPUT_ROOT}"
echo "Object eval: ${OBJECT_OUTPUT_ROOT}"