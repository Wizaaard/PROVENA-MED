#!/usr/bin/env bash
#SBATCH -J provena_mprobe
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# (M) attention-knockout probe worker: one model -> per-citation Delta^M rows.
set -euo pipefail
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${N:=60}"
: "${COHORT:=cardiac_mm}"
: "${MAXCLAIMS:=8}"

CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0
cd "$CODE"

STAGED="outputs/staged_prov_${COHORT}_${MODEL_ID}.jsonl"
OUT="outputs/m_probe_${COHORT}_${MODEL_ID}.jsonl"
[ -f "$STAGED" ] || { echo "missing $STAGED"; exit 1; }
echo "[$(date +%T)] M-probe $MODEL_ID ($COHORT, n=$N) on $(hostname)"
python -m provena_med._dispatch probe_m --model "$MODEL_NAME" --cohort "$COHORT" --staged "$STAGED" \
    --n "$N" --max-claims "$MAXCLAIMS" --out "$OUT"
echo "[$(date +%T)] DONE M-probe $MODEL_ID ($COHORT) -> $OUT"
