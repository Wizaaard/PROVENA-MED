#!/usr/bin/env bash
#SBATCH -J provena_oracle_staged
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# Staged provenance generation under the 4 oracle reveal conditions (ED, n=300).
# Produces outputs/staged_oracle_{chief_only,history,history_tests,all}_<MODEL_ID>.jsonl
# for downstream W-judge scoring (tab:res-interactive provenance columns).
set -euo pipefail
: "${MODEL_ID:=llama31_8b}"
: "${MODEL_NAME:=${PROVENA_MODELS_DIR:-$HOME/models}/Llama-3.1-8B-Instruct}"
: "${N:=300}"
: "${BS:=8}"

CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0
cd "$CODE"

echo "[$(date +%T)] oracle-staged $MODEL_ID ED (n=$N, bs=$BS) on $(hostname)"
python -m provena_med._dispatch run_oracle_staged \
    --model "$MODEL_NAME" \
    --model-id "$MODEL_ID" \
    --n "$N" \
    --seed 0 \
    --out-dir outputs \
    --batch-size "$BS"
echo "[$(date +%T)] DONE oracle-staged $MODEL_ID"
