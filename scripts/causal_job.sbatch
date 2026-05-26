#!/usr/bin/env bash
#SBATCH -J provena_causalW
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# (W) causal MM-AIS: Shapley + necessity + sufficiency over a set of staged_prov outputs.
set -euo pipefail
: "${COHORT:=cardiac_mm}"
: "${N:=40}"
: "${IDS:=gemma3_12b gemma3_27b medgemma_27b llama31_8b med42_8b mistral7b}"
CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0,1
cd "$CODE"
OUT="outputs/causalW_${COHORT}.jsonl"; : > "$OUT"
for id in $IDS; do
  f="outputs/staged_prov_${COHORT}_${id}.jsonl"
  [ -f "$f" ] || { echo "skip missing $f"; continue; }
  echo "[$(date +%T)] causal-W $id ($COHORT, n=$N)"
  python -m provena_med._dispatch causal_mmais --in "$f" --n "$N" --out "$OUT"
done
echo "[$(date +%T)] DONE causal-W $COHORT -> $OUT"; cat "$OUT" | python -c "import sys,json; [print(json.loads(l)['file'], json.loads(l)['claims_scored']) for l in sys.stdin]"
