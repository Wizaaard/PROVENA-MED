#!/usr/bin/env bash
#SBATCH -J provena_int
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# Interactive (evidence-seeking) panel worker: one model -> ED cohort, agent mode, budget=k.
# Matches the existing Llama-3.1-8B baseline reported in Table~\ref{tab:res-interactive}.
set -euo pipefail
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${N:=300}"
: "${BUDGET:=3}"
: "${BS:=8}"

CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0
cd "$CODE"

OUT="outputs/int_agent_${MODEL_ID}.jsonl"
echo "[$(date +%T)] int $MODEL_ID ED (n=$N, budget=$BUDGET, bs=$BS) on $(hostname)"
python -m provena_med._dispatch run_interactive \
    --mode agent \
    --model "$MODEL_NAME" \
    --n "$N" \
    --budget "$BUDGET" \
    --batch-size "$BS" \
    --out "$OUT"
echo "[$(date +%T)] DONE int $MODEL_ID -> $OUT"
