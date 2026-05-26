#!/usr/bin/env bash
#SBATCH -J provena_oracle_judge
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# W-judge phase for the 4 oracle-staged JSONLs (Llama-3.1-8B by default).
# Outputs: outputs/oracle_prov_<MODEL_ID>.jsonl
set -euo pipefail
: "${MODEL_ID:=llama31_8b}"
CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0,1
cd "$CODE"
echo "[$(date +%T)] oracle-judge $MODEL_ID on $(hostname)"
python -m provena_med._dispatch score_oracle_prov --model-id "$MODEL_ID"
echo "[$(date +%T)] DONE oracle-judge $MODEL_ID"
