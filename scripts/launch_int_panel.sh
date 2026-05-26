#!/usr/bin/env bash
# Submit one interactive-track (agent, ED) job per model. Args: [N] [BUDGET] [ID-subset]
set -euo pipefail
CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
M=${PROVENA_MODELS_DIR:-$HOME/models}
N="${1:-300}"
BUDGET="${2:-3}"
IDS_IN="${3:-}"
mkdir -p "$CODE/scripts/logs_staged" "$CODE/outputs"

declare -A MODELS=(
  [llama31_8b]=$M/Llama-3.1-8B-Instruct
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
# Conservative vs dx_job: agent mode grows context turn-by-turn, so big models step down.
declare -A BS=( [llama31_8b]=8 [llama32_3b]=8 [med42_8b]=8 [mistral7b]=8 [biomistral_7b]=8 [gemma3_4b]=8 [gemma3_12b]=4 [gemma3_27b]=2 [medgemma_4b]=8 [medgemma_27b]=2 )

IDS=(${IDS_IN:-${!MODELS[@]}})
for id in "${IDS[@]}"; do
  jid=$(sbatch --parsable -J "int_${id}" ${SLURM_EXCLUDE:+--exclude=$SLURM_EXCLUDE} \
        --export=ALL,MODEL_ID="${id}",MODEL_NAME="${MODELS[$id]}",N="${N}",BUDGET="${BUDGET}",BS="${BS[$id]}",CONDA_ENV=ragcon \
        "$CODE/scripts/int_job.sbatch")
  echo "submitted int $id -> job $jid"
done
echo "---"
squeue -u "$USER" -h -o "%.10i %.16j %.10T %.20R" 2>/dev/null | grep "int_" | head -20
