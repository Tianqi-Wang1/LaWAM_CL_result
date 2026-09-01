#!/usr/bin/env bash

set -euo pipefail


# ==========================================================
# 0. Environment
# ==========================================================

source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh

conda activate \
  /home/jincai_guo/tianqi/CVPR2027/envs/lawam

cd /home/jincai_guo/tianqi/CVPR2027/LaWAM


# ==========================================================
# 1. Paths
# ==========================================================

CKPT_ROOT="/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft"

OUTPUT_ROOT="results/eval_runs/lawam_cl/libero_goal/seqft_full"

mkdir -p "${OUTPUT_ROOT}"


# ==========================================================
# 2. LIBERO environment
# ==========================================================

export LIBERO_HOME=\
/home/jincai_guo/tianqi/CVPR2027/LIBERO

export LIBERO_PYTHON=\
/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python

export STAR_VLA_PYTHON=\
/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python


# ==========================================================
# 3. Evaluation resources
#
# Can be overridden at launch:
#
#   GPU_ID=2
#   EVAL_GPU_ID=3
#   NUM_WORKERS=16
#
# ==========================================================

GPU_ID="${GPU_ID:-2}"
EVAL_GPU_ID="${EVAL_GPU_ID:-3}"

NUM_WORKERS="${NUM_WORKERS:-16}"

NUM_TRIALS="${NUM_TRIALS:-50}"

# Full CL evaluation = 2000 episodes.
# Saving every video is expensive.
#
# Default is False.
#
# To save videos:
#   SAVE_VIDEOS=True bash ...
#
SAVE_VIDEOS="${SAVE_VIDEOS:-False}"


# ==========================================================
# 4. Helper: find latest formal training run
# ==========================================================

find_run() {
    local pattern="$1"

    find "${CKPT_ROOT}" \
        -maxdepth 1 \
        -type d \
        -name "${pattern}" \
        | sort \
        | tail -n 1
}


# ==========================================================
# 5. Resolve all five checkpoints
# ==========================================================

BASE_RUN=$(find_run \
    '*+base_t0_5_10k_4gpu_bs32_ga2'
)

CL1_RUN=$(find_run \
    '*+cl1_t6_2k_4gpu_bs32_ga2'
)

CL2_RUN=$(find_run \
    '*+cl2_t7_2k_4gpu_bs32_ga2'
)

CL3_RUN=$(find_run \
    '*+cl3_t8_2k_4gpu_bs32_ga2'
)

CL4_RUN=$(find_run \
    '*+cl4_t9_2k_4gpu_bs32_ga2'
)


for var_name in \
    BASE_RUN \
    CL1_RUN \
    CL2_RUN \
    CL3_RUN \
    CL4_RUN
do

    run_path="${!var_name}"

    if [ -z "${run_path}" ]; then
        echo "[ERROR] Missing run: ${var_name}"
        exit 1
    fi

    if [ ! -f \
        "${run_path}/final_model/pytorch_model.pt" \
    ]; then

        echo "[ERROR] Missing checkpoint:"
        echo \
          "${run_path}/final_model/pytorch_model.pt"

        exit 1
    fi

    if [ ! -f \
        "${run_path}/dataset_statistics.json" \
    ]; then

        echo "[ERROR] Missing statistics:"
        echo \
          "${run_path}/dataset_statistics.json"

        exit 1
    fi

done


# ==========================================================
# 6. Print checkpoint chain
# ==========================================================

echo
echo "=========================================================="
echo " LaWAM LIBERO-Goal CL Evaluation"
echo "=========================================================="

echo
echo "Base:"
echo "  ${BASE_RUN}"

echo
echo "CL1:"
echo "  ${CL1_RUN}"

echo
echo "CL2:"
echo "  ${CL2_RUN}"

echo
echo "CL3:"
echo "  ${CL3_RUN}"

echo
echo "CL4:"
echo "  ${CL4_RUN}"

echo
echo "Policy GPU       : ${GPU_ID}"
echo "Evaluator GPU    : ${EVAL_GPU_ID}"
echo "Workers          : ${NUM_WORKERS}"
echo "Trials / task    : ${NUM_TRIALS}"
echo "Save videos      : ${SAVE_VIDEOS}"

echo "=========================================================="
echo


# ==========================================================
# 7. Master output directory
# ==========================================================

MASTER_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MASTER_DIR="${OUTPUT_ROOT}/cl_full_${MASTER_TIMESTAMP}"

mkdir -p "${MASTER_DIR}"


MANIFEST="${MASTER_DIR}/eval_manifest.tsv"

printf \
  "stage\tcheckpoint_run\teval_run_dir\ttask_ids\n" \
  > "${MANIFEST}"


# ==========================================================
# 8. Helper: run one CL stage evaluation
# ==========================================================

run_eval_stage() {

    local stage="$1"
    local train_run="$2"
    local task_ids="$3"
    local alias="$4"

    local ckpt
    ckpt="${train_run}/final_model/pytorch_model.pt"

    echo
    echo
    echo "=========================================================="
    echo " Evaluating ${stage}"
    echo "=========================================================="
    echo "Checkpoint:"
    echo "  ${ckpt}"
    echo
    echo "Task IDs:"
    echo "  ${task_ids}"
    echo "=========================================================="
    echo


    # ------------------------------------------------------
    # Record directory state before launching.
    # ------------------------------------------------------

    local alias_root
    alias_root="${MASTER_DIR}/${alias}"

    mkdir -p "${alias_root}"


    # ------------------------------------------------------
    # Launch benchmark
    # ------------------------------------------------------

    SUITES="libero_goal" \
    TASK_IDS="${task_ids}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    GPU_IDS="${GPU_ID}" \
    EVAL_GPU_IDS="${EVAL_GPU_ID}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    OUTPUT_ROOT="${MASTER_DIR}" \
    LIBERO_CKPT_ALIAS="${alias}" \
    bash \
    examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh \
    "${ckpt}"


    # ------------------------------------------------------
    # Find the timestamped evaluation directory.
    #
    # Structure:
    #
    # MASTER_DIR/
    #   alias/
    #     <timestamp>/
    #       suites/
    #         libero_goal/
    # ------------------------------------------------------

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


    local suite_dir
    suite_dir="${eval_timestamp_dir}/suites/libero_goal"


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


    # ------------------------------------------------------
    # Convert string:
    #
    # "0 1 2 3"
    #
    # into Python CLI arguments.
    # ------------------------------------------------------

    read -r -a TASK_ARRAY <<< "${task_ids}"


    # ------------------------------------------------------
    # Per-task aggregation
    # ------------------------------------------------------

    echo
    echo "[INFO] Aggregating per-task results for ${stage}..."

    python \
      scripts/summarize_libero_cl_eval.py \
      --run-dir "${suite_dir}" \
      --task-ids "${TASK_ARRAY[@]}" \
      --expected-trials "${NUM_TRIALS}"


    # ------------------------------------------------------
    # Add to global manifest
    # ------------------------------------------------------

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
    echo

}


# ==========================================================
# 9. Sequential evaluation
# ==========================================================

run_eval_stage \
    "Base" \
    "${BASE_RUN}" \
    "0 1 2 3 4 5" \
    "base_t0_5_10k"


run_eval_stage \
    "CL1" \
    "${CL1_RUN}" \
    "0 1 2 3 4 5 6" \
    "cl1_t6_2k"


run_eval_stage \
    "CL2" \
    "${CL2_RUN}" \
    "0 1 2 3 4 5 6 7" \
    "cl2_t7_2k"


run_eval_stage \
    "CL3" \
    "${CL3_RUN}" \
    "0 1 2 3 4 5 6 7 8" \
    "cl3_t8_2k"


run_eval_stage \
    "CL4" \
    "${CL4_RUN}" \
    "0 1 2 3 4 5 6 7 8 9" \
    "cl4_t9_2k"


# ==========================================================
# 10. Build complete CL performance matrix
# ==========================================================

echo
echo "=========================================================="
echo " Building complete CL performance matrix"
echo "=========================================================="
echo


python - "${MANIFEST}" "${MASTER_DIR}" "${NUM_TRIALS}" <<'PY'
import csv
import json
import sys
from pathlib import Path


manifest_path = Path(sys.argv[1])
master_dir = Path(sys.argv[2])
expected_trials = int(sys.argv[3])


# ==========================================================
# Load manifest
# ==========================================================

manifest_rows = []

with manifest_path.open(
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t",
    )

    manifest_rows = list(reader)


# ==========================================================
# Load all per-task results
# ==========================================================

long_rows = []

stage_results = {}

stage_summaries = []


for item in manifest_rows:

    stage = item["stage"]

    run_dir = Path(
        item["eval_run_dir"]
    )

    per_task_path = (
        run_dir
        / "per_task_summary.json"
    )

    with per_task_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = json.load(f)


    stage_results[stage] = {}


    for row in rows:

        task_id = int(
            row["task_id"]
        )

        sr = float(
            row["success_rate"]
        )

        successes = int(
            row["successes"]
        )

        trials = int(
            row["trials"]
        )


        if trials != expected_trials:
            raise RuntimeError(
                f"{stage}/task{task_id}: "
                f"expected {expected_trials} trials, "
                f"got {trials}"
            )


        stage_results[
            stage
        ][task_id] = sr


        long_rows.append(
            {
                "stage": stage,
                "task_id": task_id,
                "task_description": (
                    row["task_description"]
                ),
                "successes": successes,
                "trials": trials,
                "success_rate": sr,
            }
        )


    total_success = sum(
        int(row["successes"])
        for row in rows
    )

    total_trials = sum(
        int(row["trials"])
        for row in rows
    )

    mean_task_sr = sum(
        float(row["success_rate"])
        for row in rows
    ) / len(rows)


    stage_summaries.append(
        {
            "stage": stage,
            "num_tasks": len(rows),
            "successes": total_success,
            "trials": total_trials,
            "overall_success_rate": (
                total_success / total_trials
            ),
            "mean_task_success_rate": (
                mean_task_sr
            ),
        }
    )


# ==========================================================
# Write long-format table
# ==========================================================

long_path = (
    master_dir
    / "cl_per_task_long.csv"
)

with long_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

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


# ==========================================================
# Write stage summary
# ==========================================================

stage_summary_path = (
    master_dir
    / "cl_stage_summary.csv"
)

with stage_summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

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
    writer.writerows(
        stage_summaries
    )


# ==========================================================
# Write CL performance matrix
# ==========================================================

stage_order = [
    "Base",
    "CL1",
    "CL2",
    "CL3",
    "CL4",
]

matrix_path = (
    master_dir
    / "cl_performance_matrix.csv"
)


with matrix_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "stage"
    ] + [
        f"task_{task_id}"
        for task_id in range(10)
    ] + [
        "mean_seen_task_sr"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()


    for stage in stage_order:

        result = stage_results[
            stage
        ]

        row = {
            "stage": stage
        }


        for task_id in range(10):

            if task_id in result:
                row[
                    f"task_{task_id}"
                ] = (
                    f"{result[task_id]:.4f}"
                )

            else:
                row[
                    f"task_{task_id}"
                ] = ""


        row[
            "mean_seen_task_sr"
        ] = (
            f"{sum(result.values()) / len(result):.4f}"
        )


        writer.writerow(row)


# ==========================================================
# Print matrix
# ==========================================================

print()
print(
    "Stage | "
    + " | ".join(
        f"T{i}"
        for i in range(10)
    )
    + " | Mean"
)

print(
    "-" * 100
)


for stage in stage_order:

    result = stage_results[
        stage
    ]

    values = []

    for task_id in range(10):

        if task_id in result:
            values.append(
                f"{result[task_id]:.2f}"
            )

        else:
            values.append(
                "-"
            )


    mean_sr = (
        sum(result.values())
        / len(result)
    )


    print(
        f"{stage:>4s} | "
        + " | ".join(
            f"{value:>4s}"
            for value in values
        )
        + f" | {mean_sr:.4f}"
    )


print()
print("[OK] Saved:")
print(matrix_path)
print(long_path)
print(stage_summary_path)

PY


# ==========================================================
# 11. Final summary
# ==========================================================

echo
echo
echo "=========================================================="
echo " All CL evaluations completed"
echo "=========================================================="

echo
echo "Master directory:"
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
echo "=========================================================="

