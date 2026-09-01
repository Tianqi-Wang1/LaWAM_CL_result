#!/usr/bin/env bash
set -euo pipefail
source /apps/miniconda3/etc/profile.d/conda.sh
source /home/jincai_guo/tianqi/CVPR2027/setup_paths.sh
conda activate /home/jincai_guo/tianqi/CVPR2027/envs/lawam

ROOT=/home/jincai_guo/tianqi/CVPR2027/LaWAM
cd "$ROOT"
BASE_ROOT=/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/seqft
EXP_ROOT=/home/jincai_guo/tianqi/CVPR2027/checkpoints/lawam_cl/libero_goal/t9_vlm_flow_lora_v5/text_all
RUN_ROOT="$EXP_ROOT/runs"
BASE_RUN=$(find "$BASE_ROOT" -maxdepth 1 -type d -name '*+base_t0_5_10k_4gpu_bs32_ga2' | sort | tail -n 1)
RUN=${RUN_OVERRIDE:-$(find "$RUN_ROOT" -maxdepth 1 -type d -name '*+t9_vlm_text_all_flow_dit16_lora_r8_2000step_4gpu_bs64_ga1' | sort | tail -n 1)}
[ -n "$BASE_RUN" ] && [ -n "$RUN" ] || { echo '[ERROR] Base/run not found'; exit 1; }
BASE="$BASE_RUN/final_model/pytorch_model.pt"
UNMERGED="$RUN/final_model/pytorch_model.pt"
MERGED="$RUN/final_model/pytorch_model_merged.pt"

python scripts/merge_vlm_flow_lora_checkpoint_v5.py --input "$UNMERGED" --output "$MERGED" --alpha 8
python scripts/verify_vlm_flow_lora_checkpoint_v5.py --base "$BASE" --unmerged "$UNMERGED" --merged "$MERGED" --variant text_all --rank 8

export LIBERO_HOME=/home/jincai_guo/tianqi/CVPR2027/LIBERO
export LIBERO_PYTHON=/home/jincai_guo/tianqi/CVPR2027/bin/libero_osmesa_python
export STAR_VLA_PYTHON=/home/jincai_guo/tianqi/CVPR2027/envs/lawam/bin/python
POLICY_GPU=${POLICY_GPU:-4}; EVAL_GPU=${EVAL_GPU:-5}; NUM_TRIALS=${NUM_TRIALS:-50}; EVAL_WORKERS=${EVAL_WORKERS:-16}
STAMP=$(date +%Y%m%d_%H%M%S)
ALIAS=t9_vlm_text_all_flow_dit16_lora_r8
OUT="$ROOT/results/eval_runs/lawam_cl/libero_goal/t9_vlm_flow_lora_v5/text_all/resume_${STAMP}"
mkdir -p "$OUT"
SUITES=libero_goal TASK_IDS=9 NUM_TRIALS_PER_TASK="$NUM_TRIALS" NUM_WORKERS="$EVAL_WORKERS" \
GPU_IDS="$POLICY_GPU" EVAL_GPU_IDS="$EVAL_GPU" SAVE_VIDEOS=False OUTPUT_ROOT="$OUT" LIBERO_CKPT_ALIAS="$ALIAS" \
bash examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh "$MERGED"
EVAL_DIR=$(find "$OUT/$ALIAS" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
SUITE_DIR="$EVAL_DIR/suites/libero_goal"
python scripts/summarize_libero_cl_eval.py --run-dir "$SUITE_DIR" --task-ids 9 --expected-trials "$NUM_TRIALS"
echo "$SUITE_DIR/per_task_summary.csv" > "$EXP_ROOT/latest_summary_path.txt"
echo "[OK] text_all resumed without retraining: $SUITE_DIR/per_task_summary.csv"
