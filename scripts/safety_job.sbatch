#!/usr/bin/env bash
#SBATCH -J provena_safety
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o ${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}/scripts/logs_safety/%x_%j.out
#SBATCH -e ${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}/scripts/logs_safety/%x_%j.err
# ---------------------------------------------------------------------------
# PROVENA-MED safety panel worker: one model per job (transformers backend).
# Required env: MODEL_ID, MODEL_NAME. Optional: N (default 300), BS (default 16).
# ---------------------------------------------------------------------------
set -euo pipefail
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${N:=300}"
: "${BS:=16}"

CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
CONDA_ENV="${CONDA_ENV:-ragcon}"

source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0
cd "$CODE"

echo "[$(date +%T)] $MODEL_ID :: $MODEL_NAME (n=$N, bs=$BS) on $(hostname)"
python -m provena_med._dispatch run_generate_safety --model "$MODEL_NAME" --n "$N" \
    --out "outputs/safety_panel_${MODEL_ID}.jsonl" --batch-size "$BS"
python -m provena_med._dispatch score_safety --in "outputs/safety_panel_${MODEL_ID}.jsonl" \
    > "outputs/score_${MODEL_ID}.txt" 2>&1
echo "[$(date +%T)] DONE $MODEL_ID"
cat "outputs/score_${MODEL_ID}.txt"
