#!/usr/bin/env bash
#SBATCH -J provena_dx
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# Differential-diagnosis panel worker: one model -> ranked differential on all 5 cohorts.
set -euo pipefail
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${N:=150}"
: "${BS:=8}"
: "${COHORTS:=ed cardiac_mm icu_mm eicu mimic3}"

CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0
cd "$CODE"

for COHORT in $COHORTS; do
  OUT="outputs/dx_${COHORT}_${MODEL_ID}.jsonl"
  echo "[$(date +%T)] dx $MODEL_ID $COHORT (n=$N, bs=$BS) on $(hostname)"
  python -m provena_med._dispatch run_generate_dx --model "$MODEL_NAME" --cohort "$COHORT" --n "$N" \
      --out "$OUT" --batch-size "$BS"
  echo "[$(date +%T)] DONE dx $MODEL_ID $COHORT -> $OUT"
done
echo "[$(date +%T)] ALL DONE dx $MODEL_ID"
