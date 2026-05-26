#!/usr/bin/env bash
# Submit one SLURM job per model for the safety panel. Args: [N] [ID-subset]
set -euo pipefail
CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
JOB="$CODE/scripts/safety_job.sbatch"
M=${PROVENA_MODELS_DIR:-$HOME/models}
N="${1:-300}"
IDS_IN="${2:-}"
mkdir -p "$CODE/scripts/logs_safety" "$CODE/outputs"

declare -A MODELS=(
  [llama31_8b]=$M/Llama-3.1-8B-Instruct
  [llama32_1b]=$M/Llama-3.2-1B-Instruct
  [llama32_3b]=$M/Llama-3.2-3B-Instruct
  [med42_8b]=$M/Llama3-Med42-8B
  [mistral7b]=$M/Mistral-7B-Instruct-v0.3
  [biomistral_7b]=$M/BioMistral-7B-DARE
  [gemma3_4b]=$M/gemma-3-4b-it
  [gemma3_12b]=$M/gemma-3-12b-it
  [gemma3_27b]=$M/gemma-3-27b-it
  [medgemma_4b]=$M/medgemma-4b-it
  [medgemma_27b]=$M/medgemma-27b-it
)
declare -A BS=( [llama31_8b]=16 [llama32_1b]=16 [llama32_3b]=16 [med42_8b]=16 [mistral7b]=16 [biomistral_7b]=16 [gemma3_4b]=16 [gemma3_12b]=8 [gemma3_27b]=6 [medgemma_4b]=16 [medgemma_27b]=6 )

IDS=(${IDS_IN:-${!MODELS[@]}})
for id in "${IDS[@]}"; do
  jid=$(sbatch --parsable -J "saf_${id}" \
        --export=ALL,MODEL_ID="${id}",MODEL_NAME="${MODELS[$id]}",N="${N}",BS="${BS[$id]}",CONDA_ENV=ragcon \
        "$JOB")
  echo "submitted $id -> job $jid"
done
echo "---"
squeue -u "$USER" -h -o "%.10i %.16j %.8T" 2>/dev/null | grep saf_ | head -20
