#!/usr/bin/env bash
#SBATCH -J provena_judgepanel
#SBATCH --qos=normal  # override per-cluster: sbatch --qos=YOURS
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH -o ./logs/%x_%j.out  # override: sbatch -o /path/to/logs/%x_%j.out
#SBATCH -e ./logs/%x_%j.err
# Judge phase: (W) causal MM-AIS + WxM quadrants for the whole model slate, one 70B load.
set -euo pipefail
: "${COHORT:=cardiac_mm}"
: "${IDS:?Need IDS}"
: "${N:=60}"
CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-ragcon}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=0,1
cd "$CODE"
echo "[$(date +%T)] judge-panel $COHORT ids=[$IDS] n=$N skip_causal=${SKIP_CAUSAL:-0} on $(hostname)"
python -m provena_med._dispatch panel_judge --cohort "$COHORT" --n "$N" --ids $IDS \
    ${SKIP_CAUSAL:+--skip-causal} \
    --causal-out "outputs/causalW_panel_${COHORT}.jsonl" \
    --quad-out "outputs/quadrants_panel_${COHORT}.jsonl"
echo "[$(date +%T)] DONE judge-panel $COHORT"
