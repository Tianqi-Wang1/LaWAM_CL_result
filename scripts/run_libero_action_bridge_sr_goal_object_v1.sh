#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LaWAM Closed-Loop Action-Bridge SR Intervention -- Goal + Object
#
# Purpose
# -------
# Behavioral bridge for the previous offline 6-route action-coupling analysis.
# Instead of fixed-anchor Flow MSE, this script measures TRUE LIBERO closed-loop
# success rate (SR) for the same interventions on historical Base tasks 0-5.
#
# At CL stage k:
#
#   base_ref       : F_B(H_VLM^B, Future_B)
#   vlm_only       : F_B(H_VLM^k, Future_B)
#   world_only     : F_B(H_VLM^B, Future_k)
#   upstream_joint : F_B(H_VLM^k, Future_k)
#   flow_only      : F_k(H_VLM^B, Future_B)
#   full           : F_k(H_VLM^k, Future_k)
#
# Exact matching detail:
#   - h_t is ALWAYS computed with the Base LAM visual encoder.
#   - Future_B uses Base VLM/queries + Base QFormer + Base LaWM decoder.
#   - Future_k uses CL-k VLM/queries + CL-k QFormer + CL-k LaWM decoder,
#     decoded from the SAME Base h_t.
#   - The online input processor/batch builder is anchored to Base for all routes.
#
# This mirrors the previous offline action-coupling intervention as closely as
# possible while replacing the local MSE probe with real closed-loop rollout SR.
#
# Formal default workload:
#   suites : libero_goal libero_object
#   stages : CL1 CL2 CL3 CL4
#   routes : all six
#   tasks  : 0 1 2 3 4 5
#   trials : 50/task
#
# Total formal episodes = 2 * 4 * 6 * 6 * 50 = 14,400.
# This is intentionally expensive. Use the smoke/focused examples below first.
#
# Recommended smoke:
#   SUITES="libero_goal" STAGES="CL4" TASK_IDS="0" NUM_TRIALS=3 \
#   EVAL_WORKERS=1 POLICY_GPUS=2 EVAL_GPU=7 \
#   bash scripts/run_libero_action_bridge_sr_goal_object_v1.sh
#
# Focused bridge (good first formal pass):
#   STAGES="CL4" NUM_TRIALS=50 POLICY_GPUS=2 EVAL_GPU=7 \
#   bash scripts/run_libero_action_bridge_sr_goal_object_v1.sh
#
# Full formal:
#   NUM_TRIALS=50 POLICY_GPUS=2 EVAL_GPU=7 \
#   bash scripts/run_libero_action_bridge_sr_goal_object_v1.sh
#
# If two policy GPUs are available (and a third GPU is available for simulation),
# the Base and CL policies can be split across GPUs to reduce per-GPU memory:
#   POLICY_GPUS=2,6 EVAL_GPU=7 bash ...
# =============================================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT="/home/jincai_guo/tianqi/CVPR2027/LaWAM"
cd "${ROOT}"

SERVER_SCRIPT="${ROOT}/deployment/model_server/server_policy_action_bridge_sr.py"
SUMMARY_SCRIPT="${ROOT}/scripts/summarize_libero_cl_eval.py"
COMMON_SCRIPT="${ROOT}/examples/LIBERO/eval_files/auto_eval_scripts/common.sh"
EVAL_SCRIPT="${ROOT}/examples/LIBERO/eval_files/eval_libero.py"

for required in \
    "${SERVER_SCRIPT}" \
    "${SUMMARY_SCRIPT}" \
    "${COMMON_SCRIPT}" \
    "${EVAL_SCRIPT}"
do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Missing required file:"
        echo "        ${required}"
        exit 1
    fi
done

# Reuse the validated port-wait helper.
# shellcheck source=/dev/null
source "${COMMON_SCRIPT}"

# =============================================================================
# Experiment controls
# =============================================================================

SUITES="${SUITES:-libero_goal libero_object}"
STAGES="${STAGES:-CL1 CL2 CL3 CL4}"
ROUTES="${ROUTES:-base_ref vlm_only world_only upstream_joint flow_only full}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5}"

NUM_TRIALS="${NUM_TRIALS:-50}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"
EVAL_SEED="${EVAL_SEED:-0}"
POLICY_SEED="${POLICY_SEED:-2026}"

# One policy GPU is sufficient if two LaWAM policies fit in memory.
# Two policy GPUs are also supported: Base -> cuda:0, CL -> cuda:1.
POLICY_GPUS="${POLICY_GPUS:-${POLICY_GPU:-2}}"
EVAL_GPU="${EVAL_GPU:-7}"

SERVER_STARTUP_TIMEOUT_SEC="${SERVER_STARTUP_TIMEOUT_SEC:-900}"
WORKER_RESULT_TIMEOUT_SEC="${WORKER_RESULT_TIMEOUT_SEC:-600}"
WORKER_SYNC_TIMEOUT_SEC="${WORKER_SYNC_TIMEOUT_SEC:-1.0}"
PORT_SEARCH_START="${PORT_SEARCH_START:-6694}"
PORT_SEARCH_LIMIT="${PORT_SEARCH_LIMIT:-1200}"

GOAL_RUN_ROOT="${GOAL_RUN_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft}"
OBJECT_RUN_ROOT="${OBJECT_RUN_ROOT:-/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_object/seqft}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/eval_runs/lawam_cl/action_bridge_sr}"
MASTER_DIR="${OUTPUT_ROOT}/bridge_${TIMESTAMP}"
mkdir -p "${MASTER_DIR}"

MANIFEST="${MASTER_DIR}/eval_manifest.tsv"
printf "suite\tstage\troute\tbase_run\tstage_run\teval_dir\tport\ttask_ids\n" > "${MANIFEST}"

MASTER_LOG="${MASTER_DIR}/action_bridge_sr.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export STARVLA_WORKER_OMP_THREADS=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset NCCL_TOPO_FILE || true
unset NCCL_GRAPH_FILE || true
unset NCCL_CONF_FILE || true
unset HFAI_NCCL_OPT_LEVEL || true

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
export PYTHONPATH="${LIBERO_HOME}:${ROOT}:${PYTHONPATH:-}"

# =============================================================================
# Validation helpers
# =============================================================================

canonical_route_offset() {
    case "$1" in
        base_ref)       echo 0 ;;
        vlm_only)       echo 1 ;;
        world_only)     echo 2 ;;
        upstream_joint) echo 3 ;;
        flow_only)      echo 4 ;;
        full)           echo 5 ;;
        *)
            echo "[ERROR] Unsupported route: $1" >&2
            exit 1
            ;;
    esac
}

stage_number() {
    case "$1" in
        CL1) echo 1 ;;
        CL2) echo 2 ;;
        CL3) echo 3 ;;
        CL4) echo 4 ;;
        *)
            echo "[ERROR] STAGES supports CL1 CL2 CL3 CL4 only; got: $1" >&2
            exit 1
            ;;
    esac
}

find_run() {
    local root="$1"
    local pattern="$2"
    find "${root}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -n 1
}

verify_run() {
    local label="$1"
    local run="$2"
    if [ -z "${run}" ]; then
        echo "[ERROR] ${label}: run not found."
        exit 1
    fi
    for required in \
        "${run}/config.yaml" \
        "${run}/dataset_statistics.json" \
        "${run}/final_model/pytorch_model.pt"
    do
        if [ ! -f "${required}" ]; then
            echo "[ERROR] ${label}: missing ${required}"
            exit 1
        fi
    done
    echo "[OK] ${label}: ${run}"
}

resolve_chain_for_suite() {
    local suite="$1"
    local root="$2"

    BASE_RUN=$(find_run "${root}" '*+base_t0_5_10k_4gpu_bs32_ga2')
    CL1_RUN=$(find_run "${root}" '*+cl1_t6_2k_4gpu_bs32_ga2')
    CL2_RUN=$(find_run "${root}" '*+cl2_t7_2k_4gpu_bs32_ga2')
    CL3_RUN=$(find_run "${root}" '*+cl3_t8_2k_4gpu_bs32_ga2')
    CL4_RUN=$(find_run "${root}" '*+cl4_t9_2k_4gpu_bs32_ga2')

    verify_run "${suite} Base" "${BASE_RUN}"
    verify_run "${suite} CL1" "${CL1_RUN}"
    verify_run "${suite} CL2" "${CL2_RUN}"
    verify_run "${suite} CL3" "${CL3_RUN}"
    verify_run "${suite} CL4" "${CL4_RUN}"

    # The client always anchors normalization to Base. Assert action/state stats
    # are exactly identical at every CL stage before running the intervention.
    "${STAR_VLA_PYTHON}" - \
        "${BASE_RUN}/dataset_statistics.json" \
        "${CL1_RUN}/dataset_statistics.json" \
        "${CL2_RUN}/dataset_statistics.json" \
        "${CL3_RUN}/dataset_statistics.json" \
        "${CL4_RUN}/dataset_statistics.json" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(x) for x in sys.argv[1:]]
with paths[0].open("r", encoding="utf-8") as f:
    ref = json.load(f)
for path in paths[1:]:
    with path.open("r", encoding="utf-8") as f:
        cur = json.load(f)
    for tag in ref:
        if tag not in cur:
            raise RuntimeError(f"{path}: missing embodiment tag {tag}")
        for section in ("action", "state"):
            if ref[tag][section] != cur[tag][section]:
                raise RuntimeError(
                    f"Normalization mismatch: {path} tag={tag} section={section}"
                )
print("[OK] Base/CL action+state normalization statistics are identical.")
PY

    # The bridge intentionally uses Base h_t for every route, matching the
    # previous offline protocol. Verify that the frozen LAM side outside the
    # trainable decoder is bitwise identical, so this does not alter native
    # stage inference semantics.
    "${STAR_VLA_PYTHON}" - \
        "${BASE_RUN}/final_model/pytorch_model.pt" \
        "${CL1_RUN}/final_model/pytorch_model.pt" \
        "${CL2_RUN}/final_model/pytorch_model.pt" \
        "${CL3_RUN}/final_model/pytorch_model.pt" \
        "${CL4_RUN}/final_model/pytorch_model.pt" <<'PY'
import sys
import torch

paths = sys.argv[1:]

def load(path):
    kwargs = dict(map_location="cpu")
    try:
        obj = torch.load(path, weights_only=True, mmap=True, **kwargs)
    except TypeError:
        try:
            obj = torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            obj = torch.load(path, **kwargs)
    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "module"):
            nested = obj.get(wrapper)
            if isinstance(nested, dict) and any(torch.is_tensor(v) for v in nested.values()):
                obj = nested
                break
    if obj and all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    return obj

def fixed_lam(state):
    prefix = "policy_backend.lam."
    decoder = "policy_backend.lam.decoder."
    return {
        k: v
        for k, v in state.items()
        if k.startswith(prefix) and not k.startswith(decoder) and torch.is_tensor(v)
    }

base_state = load(paths[0])
base = fixed_lam(base_state)
del base_state
if not base:
    raise RuntimeError("No non-decoder policy_backend.lam.* tensors found in Base checkpoint")
base_keys = set(base)
for path in paths[1:]:
    cur_state = load(path)
    cur = fixed_lam(cur_state)
    del cur_state
    if set(cur) != base_keys:
        missing = sorted(base_keys - set(cur))[:8]
        extra = sorted(set(cur) - base_keys)[:8]
        raise RuntimeError(
            f"Frozen LAM non-decoder key mismatch for {path}: missing={missing}, extra={extra}"
        )
    changed = [
        k for k in sorted(base_keys)
        if base[k].shape != cur[k].shape
        or base[k].dtype != cur[k].dtype
        or not torch.equal(base[k], cur[k])
    ]
    if changed:
        raise RuntimeError(
            f"Frozen LAM non-decoder tensors changed for {path}: "
            f"count={len(changed)}, examples={changed[:12]}"
        )
    print(f"[OK] frozen LAM non-decoder exact vs Base: {path} tensors={len(base_keys)}")
    del cur
PY
}

get_stage_run() {
    case "$1" in
        CL1) echo "${CL1_RUN}" ;;
        CL2) echo "${CL2_RUN}" ;;
        CL3) echo "${CL3_RUN}" ;;
        CL4) echo "${CL4_RUN}" ;;
        *) echo "[ERROR] Unknown stage: $1" >&2; exit 1 ;;
    esac
}

find_free_port_block() {
    "${STAR_VLA_PYTHON}" - "${PORT_SEARCH_START}" "${PORT_SEARCH_LIMIT}" <<'PY'
import socket
import sys

start = int(sys.argv[1])
limit = int(sys.argv[2])
width = 6

def free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()

for base in range(start, start + limit):
    if all(free(base + i) for i in range(width)):
        print(base)
        raise SystemExit(0)
raise SystemExit("No free 6-port block found")
PY
}

# =============================================================================
# GPU mapping
# =============================================================================

POLICY_GPUS_NORMALIZED="${POLICY_GPUS//,/ }"
read -r -a POLICY_GPU_ARRAY <<< "${POLICY_GPUS_NORMALIZED}"
if [ "${#POLICY_GPU_ARRAY[@]}" -lt 1 ] || [ "${#POLICY_GPU_ARRAY[@]}" -gt 2 ]; then
    echo "[ERROR] POLICY_GPUS must contain one or two physical GPU IDs."
    echo "        got: ${POLICY_GPUS}"
    exit 1
fi

for gpu in "${POLICY_GPU_ARRAY[@]}"; do
    if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid POLICY_GPUS entry: ${gpu}"
        exit 1
    fi
    if [ "${gpu}" = "${EVAL_GPU}" ]; then
        echo "[ERROR] EVAL_GPU=${EVAL_GPU} overlaps POLICY_GPUS=${POLICY_GPUS}."
        echo "        Use a separate simulator GPU for this dual-policy server."
        exit 1
    fi
done

BASE_DEVICE="cuda:0"
if [ "${#POLICY_GPU_ARRAY[@]}" -eq 2 ]; then
    STAGE_DEVICE="cuda:1"
else
    STAGE_DEVICE="cuda:0"
fi

# Validate requested routes/stages now, before any model load.
read -r -a ROUTE_ARRAY <<< "${ROUTES}"
read -r -a STAGE_ARRAY <<< "${STAGES}"
read -r -a TASK_ARRAY <<< "${TASK_IDS}"
read -r -a SUITE_ARRAY <<< "${SUITES}"

for route in "${ROUTE_ARRAY[@]}"; do
    canonical_route_offset "${route}" >/dev/null
done
for stage in "${STAGE_ARRAY[@]}"; do
    stage_number "${stage}" >/dev/null
done

if [ "${#TASK_ARRAY[@]}" -eq 0 ]; then
    echo "[ERROR] TASK_IDS is empty."
    exit 1
fi

# =============================================================================
# Runtime cleanup
# =============================================================================

CURRENT_SERVER_PID=""
cleanup_server() {
    if [ -n "${CURRENT_SERVER_PID}" ] && kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
        echo "[cleanup] stopping action-bridge server PID=${CURRENT_SERVER_PID}"
        kill "${CURRENT_SERVER_PID}" 2>/dev/null || true
        wait "${CURRENT_SERVER_PID}" 2>/dev/null || true
    fi
    CURRENT_SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

# =============================================================================
# One suite/stage: load Base + CL once, serve all requested routes on six ports,
# then run the LIBERO evaluator sequentially for each route.
# =============================================================================

run_suite_stage() {
    local suite="$1"
    local stage="$2"
    local base_run="$3"
    local stage_run="$4"

    local base_ckpt="${base_run}/final_model/pytorch_model.pt"
    local stage_ckpt="${stage_run}/final_model/pytorch_model.pt"
    local stage_dir="${MASTER_DIR}/${suite}/${stage}"
    mkdir -p "${stage_dir}"

    local port_base
    port_base=$(find_free_port_block)
    # Move the search start so the next stage naturally looks elsewhere.
    PORT_SEARCH_START=$((port_base + 10))

    local server_log="${stage_dir}/bridge_server.log"

    echo
    echo "================================================================================"
    echo " Closed-loop action bridge: ${suite} / ${stage}"
    echo "================================================================================"
    echo "Base run      : ${base_run}"
    echo "Stage run     : ${stage_run}"
    echo "Policy GPUs   : ${POLICY_GPUS}"
    echo "Base device   : ${BASE_DEVICE}"
    echo "Stage device  : ${STAGE_DEVICE}"
    echo "Simulator GPU : ${EVAL_GPU}"
    echo "Routes        : ${ROUTES}"
    echo "Tasks         : ${TASK_IDS}"
    echo "Trials/task   : ${NUM_TRIALS}"
    echo "Workers       : ${EVAL_WORKERS}"
    echo "Port base     : ${port_base}"
    echo "================================================================================"

    cleanup_server

    CUDA_VISIBLE_DEVICES="${POLICY_GPUS}" \
    "${STAR_VLA_PYTHON}" "${SERVER_SCRIPT}" \
        --base_ckpt_path "${base_ckpt}" \
        --stage_ckpt_path "${stage_ckpt}" \
        --port_base "${port_base}" \
        --routes "${ROUTE_ARRAY[@]}" \
        --use_bf16 \
        --base_device "${BASE_DEVICE}" \
        --stage_device "${STAGE_DEVICE}" \
        --seed "${POLICY_SEED}" \
        --idle_timeout -1 \
        > "${server_log}" 2>&1 &
    CURRENT_SERVER_PID=$!

    echo "[server] PID=${CURRENT_SERVER_PID}; waiting for route ports..."

    for route in "${ROUTE_ARRAY[@]}"; do
        local offset
        offset=$(canonical_route_offset "${route}")
        local port=$((port_base + offset))

        if ! wait_for_port \
            "${STAR_VLA_PYTHON}" \
            "127.0.0.1" \
            "${port}" \
            "${SERVER_STARTUP_TIMEOUT_SEC}"
        then
            echo "[ERROR] Bridge server route ${route} failed to become ready on port ${port}."
            echo "[ERROR] Server log: ${server_log}"
            tail -n 80 "${server_log}" || true
            exit 1
        fi
        echo "[OK] route=${route} ready on port=${port}"
    done

    for route in "${ROUTE_ARRAY[@]}"; do
        local offset
        offset=$(canonical_route_offset "${route}")
        local port=$((port_base + offset))
        local eval_dir="${stage_dir}/${route}"
        mkdir -p "${eval_dir}"

        echo
        echo "--------------------------------------------------------------------------"
        echo "[EVAL] suite=${suite} stage=${stage} route=${route} port=${port}"
        echo "       tasks=${TASK_IDS} trials/task=${NUM_TRIALS}"
        echo "--------------------------------------------------------------------------"

        local -a eval_cmd=(
            "${LIBERO_PYTHON}" "${EVAL_SCRIPT}"
            --args.pretrained-path "${base_ckpt}"
            --args.host 127.0.0.1
            --args.port "${port}"
            --args.task-suite-name "${suite}"
            --args.num-trials-per-task "${NUM_TRIALS}"
            --args.num-workers "${EVAL_WORKERS}"
            --args.worker-sync-timeout-sec "${WORKER_SYNC_TIMEOUT_SEC}"
            --args.worker-result-timeout-sec "${WORKER_RESULT_TIMEOUT_SEC}"
            --args.video-out-path "${eval_dir}"
            --args.seed "${EVAL_SEED}"
            --args.benchmark-variant libero
            --args.task-ids "${TASK_ARRAY[@]}"
        )

        if [ "${SAVE_VIDEOS}" = "True" ]; then
            eval_cmd+=(--args.save-videos)
        else
            eval_cmd+=(--args.no-save-videos)
        fi
        eval_cmd+=(--args.no-save-only-failure-videos)
        eval_cmd+=(--args.no-save-similarity-video)

        set +e
        PYTHONFAULTHANDLER=1 \
        CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
        MUJOCO_EGL_DEVICE_ID="${EVAL_GPU}" \
        MUJOCO_GL=osmesa \
        PYOPENGL_PLATFORM=osmesa \
        "${eval_cmd[@]}" 2>&1 | tee "${eval_dir}/eval.log"
        local eval_status=${PIPESTATUS[0]}
        set -e

        if [ "${eval_status}" -ne 0 ]; then
            echo "[ERROR] Evaluation failed: ${suite}/${stage}/${route} code=${eval_status}"
            echo "[ERROR] Bridge server tail:"
            tail -n 80 "${server_log}" || true
            exit "${eval_status}"
        fi

        for required in "${eval_dir}/episodes.jsonl" "${eval_dir}/summary.json"; do
            if [ ! -f "${required}" ]; then
                echo "[ERROR] Missing evaluation output: ${required}"
                exit 1
            fi
        done

        "${STAR_VLA_PYTHON}" "${SUMMARY_SCRIPT}" \
            --run-dir "${eval_dir}" \
            --task-ids "${TASK_ARRAY[@]}" \
            --expected-trials "${NUM_TRIALS}"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${suite}" \
            "${stage}" \
            "${route}" \
            "${base_run}" \
            "${stage_run}" \
            "${eval_dir}" \
            "${port}" \
            "${TASK_IDS}" \
            >> "${MANIFEST}"

        echo "[OK] ${suite}/${stage}/${route} completed."
    done

    cleanup_server
}

# =============================================================================
# Main experiment
# =============================================================================

echo "================================================================================"
echo " LaWAM Closed-Loop Action-Bridge SR Intervention"
echo "================================================================================"
echo "Suites             : ${SUITES}"
echo "Stages             : ${STAGES}"
echo "Routes             : ${ROUTES}"
echo "Historical tasks   : ${TASK_IDS}"
echo "Trials/task        : ${NUM_TRIALS}"
echo "Eval workers       : ${EVAL_WORKERS}"
echo "Policy GPU(s)      : ${POLICY_GPUS}"
echo "Simulator GPU      : ${EVAL_GPU}"
echo "Environment seed   : ${EVAL_SEED}"
echo "Policy RNG seed    : ${POLICY_SEED}"
echo "Output             : ${MASTER_DIR}"
echo "================================================================================"

for suite in "${SUITE_ARRAY[@]}"; do
    case "${suite}" in
        libero_goal)
            run_root="${GOAL_RUN_ROOT}"
            ;;
        libero_object)
            run_root="${OBJECT_RUN_ROOT}"
            ;;
        *)
            echo "[ERROR] Unsupported suite: ${suite}"
            exit 1
            ;;
    esac

    echo
    echo "################################################################################"
    echo "# Resolving vanilla SeqFT chain: ${suite}"
    echo "################################################################################"
    resolve_chain_for_suite "${suite}" "${run_root}"

    for stage in "${STAGE_ARRAY[@]}"; do
        stage_run=$(get_stage_run "${stage}")
        run_suite_stage "${suite}" "${stage}" "${BASE_RUN}" "${stage_run}"
    done
done

# =============================================================================
# Aggregate SR + paired route effects
# =============================================================================

"${STAR_VLA_PYTHON}" - "${MANIFEST}" "${MASTER_DIR}" "${NUM_TRIALS}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

manifest_path = Path(sys.argv[1])
master_dir = Path(sys.argv[2])
expected_trials = int(sys.argv[3])

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = list(csv.DictReader(f, delimiter="\t"))
if not manifest:
    raise RuntimeError("Empty bridge evaluation manifest")

canonical_routes = [
    "base_ref",
    "vlm_only",
    "world_only",
    "upstream_joint",
    "flow_only",
    "full",
]
canonical_stages = ["CL1", "CL2", "CL3", "CL4"]

long_rows = []
episode_lookup = {}
macro_lookup = {}

for row in manifest:
    suite = row["suite"]
    stage = row["stage"]
    route = row["route"]
    eval_dir = Path(row["eval_dir"])
    episode_path = eval_dir / "episodes.jsonl"

    records = []
    with episode_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_task = defaultdict(list)
    for rec in records:
        by_task[int(rec["task_id"])].append(rec)

    requested_tasks = [int(x) for x in row["task_ids"].split()]
    task_srs = []
    for task in requested_tasks:
        recs = sorted(by_task[task], key=lambda x: int(x["episode_idx"]))
        if len(recs) != expected_trials:
            raise RuntimeError(
                f"{suite}/{stage}/{route}/task{task}: "
                f"episodes={len(recs)}, expected={expected_trials}"
            )
        successes = sum(int(bool(x["success"])) for x in recs)
        sr = successes / expected_trials
        task_srs.append(sr)
        long_rows.append({
            "suite": suite,
            "stage": stage,
            "route": route,
            "task_id": task,
            "successes": successes,
            "trials": expected_trials,
            "success_rate": sr,
            "eval_dir": str(eval_dir),
        })
        for rec in recs:
            key = (suite, stage, route, task, int(rec["episode_idx"]))
            episode_lookup[key] = bool(rec["success"])

    macro_lookup[(suite, stage, route)] = statistics.fmean(task_srs)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


write_csv(
    master_dir / "bridge_sr_long.csv",
    long_rows,
    ["suite", "stage", "route", "task_id", "successes", "trials", "success_rate", "eval_dir"],
)

# Macro route/stage table.
macro_rows = []
for (suite, stage, route), sr in sorted(
    macro_lookup.items(),
    key=lambda x: (
        x[0][0],
        canonical_stages.index(x[0][1]) if x[0][1] in canonical_stages else 99,
        canonical_routes.index(x[0][2]) if x[0][2] in canonical_routes else 99,
    ),
):
    macro_rows.append({
        "suite": suite,
        "stage": stage,
        "route": route,
        "macro_success_rate": sr,
    })
write_csv(
    master_dir / "bridge_route_stage_macro.csv",
    macro_rows,
    ["suite", "stage", "route", "macro_success_rate"],
)

# Per-route task x stage matrices.
suites = sorted({r["suite"] for r in manifest})
for suite in suites:
    suite_rows = [x for x in long_rows if x["suite"] == suite]
    suite_tasks = sorted({int(x["task_id"]) for x in suite_rows})
    suite_stages = [s for s in canonical_stages if any(x["stage"] == s for x in suite_rows)]
    suite_routes = [r for r in canonical_routes if any(x["route"] == r for x in suite_rows)]
    for route in suite_routes:
        lookup = {
            (x["stage"], int(x["task_id"])): float(x["success_rate"])
            for x in suite_rows
            if x["route"] == route
        }
        rows = []
        for task in suite_tasks:
            out = {"task": f"task_{task}"}
            for stage in suite_stages:
                out[stage] = lookup.get((stage, task), "")
            rows.append(out)
        mean = {"task": "macro_mean"}
        for stage in suite_stages:
            vals = [lookup[(stage, t)] for t in suite_tasks if (stage, t) in lookup]
            mean[stage] = statistics.fmean(vals) if vals else ""
        rows.append(mean)
        write_csv(
            master_dir / suite / f"matrix_{route}_success_rate.csv",
            rows,
            ["task", *suite_stages],
        )

# Macro SR drops relative to Base reference.
effect_rows = []
for suite in suites:
    suite_stages = [s for s in canonical_stages if (suite, s, "base_ref") in macro_lookup]
    for stage in suite_stages:
        base_sr = macro_lookup[(suite, stage, "base_ref")]
        row = {
            "suite": suite,
            "stage": stage,
            "base_ref_sr": base_sr,
        }
        for route in canonical_routes[1:]:
            if (suite, stage, route) not in macro_lookup:
                continue
            sr = macro_lookup[(suite, stage, route)]
            row[f"{route}_sr"] = sr
            row[f"{route}_delta_vs_base"] = sr - base_sr
            row[f"{route}_drop_vs_base"] = base_sr - sr
        if all((suite, stage, r) in macro_lookup for r in ("vlm_only", "world_only", "upstream_joint")):
            vlm_drop = base_sr - macro_lookup[(suite, stage, "vlm_only")]
            world_drop = base_sr - macro_lookup[(suite, stage, "world_only")]
            joint_drop = base_sr - macro_lookup[(suite, stage, "upstream_joint")]
            row["upstream_interaction_drop"] = joint_drop - vlm_drop - world_drop
        effect_rows.append(row)

all_effect_fields = ["suite", "stage", "base_ref_sr"]
for route in canonical_routes[1:]:
    all_effect_fields += [
        f"{route}_sr",
        f"{route}_delta_vs_base",
        f"{route}_drop_vs_base",
    ]
all_effect_fields += ["upstream_interaction_drop"]
for row in effect_rows:
    for field in all_effect_fields:
        row.setdefault(field, "")
write_csv(master_dir / "bridge_stage_effects.csv", effect_rows, all_effect_fields)

# Paired episode-level route effects. Evaluator uses the same seed, task initial
# states, and episode indices across routes, so base_ref vs route is paired on the
# environment initialization.
paired_rows = []
for suite in suites:
    stages = [s for s in canonical_stages if any(r["suite"] == suite and r["stage"] == s for r in manifest)]
    tasks = sorted({int(x["task_id"]) for x in long_rows if x["suite"] == suite})
    for stage in stages:
        for route in canonical_routes[1:]:
            if not any(r["suite"] == suite and r["stage"] == stage and r["route"] == route for r in manifest):
                continue
            for task in tasks:
                pairs = []
                for ep in range(expected_trials):
                    kb = (suite, stage, "base_ref", task, ep)
                    kr = (suite, stage, route, task, ep)
                    if kb in episode_lookup and kr in episode_lookup:
                        pairs.append((episode_lookup[kb], episode_lookup[kr]))
                if not pairs:
                    continue
                base_success = sum(int(a) for a, _ in pairs)
                route_success = sum(int(b) for _, b in pairs)
                lost = sum(int(a and not b) for a, b in pairs)
                gained = sum(int((not a) and b) for a, b in pairs)
                both_success = sum(int(a and b) for a, b in pairs)
                both_fail = sum(int((not a) and (not b)) for a, b in pairs)
                paired_rows.append({
                    "suite": suite,
                    "stage": stage,
                    "route": route,
                    "task_id": task,
                    "paired_episodes": len(pairs),
                    "base_ref_successes": base_success,
                    "route_successes": route_success,
                    "base_ref_sr": base_success / len(pairs),
                    "route_sr": route_success / len(pairs),
                    "delta_sr": (route_success - base_success) / len(pairs),
                    "lost_successes": lost,
                    "gained_successes": gained,
                    "both_success": both_success,
                    "both_fail": both_fail,
                })

write_csv(
    master_dir / "bridge_paired_episode_effects.csv",
    paired_rows,
    [
        "suite", "stage", "route", "task_id", "paired_episodes",
        "base_ref_successes", "route_successes", "base_ref_sr", "route_sr",
        "delta_sr", "lost_successes", "gained_successes", "both_success", "both_fail",
    ],
)

print("[OK] Aggregated closed-loop bridge SR results:")
for name in (
    "bridge_sr_long.csv",
    "bridge_route_stage_macro.csv",
    "bridge_stage_effects.csv",
    "bridge_paired_episode_effects.csv",
):
    print("    ", master_dir / name)
PY

echo
echo "================================================================================"
echo " Closed-loop Action-Bridge SR experiment completed"
echo "================================================================================"
echo "Output root:"
echo "  ${MASTER_DIR}"
echo ""
echo "Core outputs:"
echo "  ${MASTER_DIR}/bridge_sr_long.csv"
echo "  ${MASTER_DIR}/bridge_route_stage_macro.csv"
echo "  ${MASTER_DIR}/bridge_stage_effects.csv"
echo "  ${MASTER_DIR}/bridge_paired_episode_effects.csv"
echo ""
echo "Per-suite route matrices:"
echo "  ${MASTER_DIR}/libero_goal/matrix_<route>_success_rate.csv"
echo "  ${MASTER_DIR}/libero_object/matrix_<route>_success_rate.csv"
echo ""
echo "Master log:"
echo "  ${MASTER_LOG}"
echo "================================================================================"