#!/usr/bin/env bash
#SBATCH -J provena_staged
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# Staged provenance panel worker: one model -> staged cited generation + MM-AIS.
set -euo pipefail
: "${MODEL_ID:?Need MODEL_ID}"
: "${MODEL_NAME:?Need MODEL_NAME}"
: "${N:=50}"
: "${BS:=8}"
: "${COHORT:=ed}"

CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0
cd "$CODE"

OUT="outputs/staged_prov_${COHORT}_${MODEL_ID}.jsonl"
echo "[$(date +%T)] $MODEL_ID staged-gen --provena cohort=$COHORT (n=$N, bs=$BS) on $(hostname)"
python -m provena_med._dispatch run_generate_staged --provena --model "$MODEL_NAME" --cohort "$COHORT" --n "$N" \
    --out "$OUT" --batch-size "$BS"
# fast objective recall sanity; precision/MM-AIS via the 70B judge is a separate scoring phase
python -c "import json; from provena_med.core.mmais import score_recall; \
r=score_recall([json.loads(l) for l in open('$OUT')]); \
print('[recall] $MODEL_ID $COHORT overall=%.3f %s'%(r['recall'], r['recall_by_modality']))"
echo "[$(date +%T)] DONE $MODEL_ID ($COHORT)"
