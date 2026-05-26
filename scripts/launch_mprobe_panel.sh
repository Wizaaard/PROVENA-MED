#!/usr/bin/env bash
# Submit one M-probe (attention-knockout) job per model, then a single judge-phase job
# (causal-W + WxM quadrants, one 70B load) that runs after them. Args: [N] [cohort] [ID-subset]
set -euo pipefail
CODE=${PROVENA_ROOT:?set PROVENA_ROOT to your provena-med checkout}
M=${PROVENA_MODELS_DIR:-$HOME/models}
N="${1:-60}"
COHORT="${2:-cardiac_mm}"
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

IDS=(${IDS_IN:-${!MODELS[@]}})
DEPS=""
for id in "${IDS[@]}"; do
  jid=$(sbatch --parsable -J "mprobe_${id}" \
        --export=ALL,MODEL_ID="${id}",MODEL_NAME="${MODELS[$id]}",N="${N}",COHORT="${COHORT}",CONDA_ENV=ragcon \
        "$CODE/scripts/mprobe_job.sbatch")
  echo "submitted M-probe $id ($COHORT) -> job $jid"
  DEPS="${DEPS}:${jid}"
done

# judge phase: causal-W (from staged) + quadrants (from m_probe), one 70B load, after all probes
jjid=$(sbatch --parsable -J "judgepanel_${COHORT}" --dependency=afterany${DEPS} \
       --export=ALL,COHORT="${COHORT}",IDS="${IDS[*]}",N="${N}",CONDA_ENV=ragcon \
       "$CODE/scripts/judge_panel_job.sbatch")
echo "submitted judge-phase ($COHORT) -> job $jjid (afterany${DEPS})"
echo "---"
squeue -u "$USER" -h -o "%.10i %.20j %.10T %.12R" 2>/dev/null | grep -E "mprobe_|judgepanel_" | head -30
