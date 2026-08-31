# Reproducing the PROVENA-MED paper

This document walks the full pipeline from raw PhysioNet downloads to the
consolidated leaderboard reported in the paper.

## 0. Prerequisites

You will need credentialed PhysioNet access (signed DUA) to each of:

| variable          | resource                                                                 |
|-------------------|--------------------------------------------------------------------------|
| `MIMIC_IV_DIR`    | MIMIC-IV (hosp/ + icu/)                                                  |
| `MIMIC_IV_ED_DIR` | MIMIC-IV-ED (medrecon, vitalsign, edstays)                               |
| `MIMIC_IV_CDS_DIR`| MIMIC-IV-Ext clinical-decision-support cohort                            |
| `MIMIC_IV_CARDIAC_DIR` | MIMIC-IV-Ext cardiac-disease cohort                                 |
| `MIMIC_CXR_DIR`   | MIMIC-CXR-JPG (or MIMIC-CXR for DICOM) + the report file                 |
| `MIMIC_III_DIR`   | MIMIC-III 1.4                                                            |
| `EICU_DIR`        | eICU-CRD 2.0                                                             |

Hardware: the W and Hit@k axes need 1 H100/H200-class GPU (judge load) for a few
hours per cohort sweep; the M (attention-knockout) axis needs an additional
GPU pass per target model. Everything fits in 80 GB VRAM.

Software:
```bash
git clone <this-repo>
cd provena-med
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vision]"   # vision/ for the DICOM pixel pipeline; omit if text-only

# Directory containing your local, credentialed Datasets/ and PROVENA-MED/ trees.
export PROVENA_DATA_ROOT=/path/to/data-root
```

## 1. Build the five cohorts

Each cohort builder reads from its corresponding PhysioNet root and writes a
single JSONL of typed bundles to `./data/cohorts/`.

```bash
# ED (primary cohort; ~9k encounters; interactive home)
provena-med build ed_cardiac --mimic-iv-ed-cds "$MIMIC_IV_CDS_DIR" \
    --mimic-iv-cardiac "$MIMIC_IV_CARDIAC_DIR" --out ./data

# ICU (~4k encounters; pairs MIMIC-IV ICU with MIMIC-CXR pixels)
provena-med build icu_mm --mimic-iv "$MIMIC_IV_DIR" \
    --mimic-cxr "$MIMIC_CXR_DIR" --out ./data

# external cohorts (different health system; different ICD era)
provena-med build eicu   --eicu      "$EICU_DIR"      --out ./data
provena-med build mimic3 --mimic-iii "$MIMIC_III_DIR" --out ./data
```

## 2. Enrich (timestamped labs, home meds, demographics, CXR pixel findings)

The base builders produce a minimal bundle. The enrichers add full-stay labs,
pre-admission medications, demographics, and (for `icu_mm`) pixel-level CXR
findings extracted by a vision model.

```bash
provena-med build labs_ts      --data ./data
provena-med build meds         --data ./data
provena-med build demographics --data ./data
provena-med build cxr_findings --cohort icu_mm --mimic-cxr "$MIMIC_CXR_DIR" \
    --out ./data/pixel_findings/icu_pixel_findings.jsonl
```

## 3. Build the guideline rule library and freeze the evaluation split

```bash
provena-med build rules --out ./data/guidelines/guideline_catalog.jsonl
provena-med build split --data ./data --out ./data/eval_split
provena-med build pack  --data ./data --out ./data/PROVENA-MED-v0.3
```

The frozen split is content-hashed; reviewers will produce byte-identical splits.

## 4. Run the open-weight panel

Pick a model (HuggingFace id; `provena_med/baselines/panel.py` lists the ten from
the paper) and run each task. Examples below use one model and one cohort; the
SLURM templates in `scripts/launch_*.sh` fan this out across the full panel.

```bash
# (a) Staged generation: produces the cited reasoning artifact
provena-med generate staged \
    --provena --cohort ed --n 150 \
    --model google/gemma-3-27b-it \
    --out outputs/staged_ed_gemma3_27b.jsonl

# (b) Diagnosis-only (focused differential prompt -> reliable Hit@k)
provena-med generate diagnosis \
    --cohort ed --n 150 \
    --model google/gemma-3-27b-it \
    --out outputs/dx_ed_gemma3_27b.jsonl

# (c) Safety (full-info ED): proposes meds; scored against the rule engine
provena-med generate safety \
    --n 300 \
    --model google/gemma-3-27b-it \
    --out outputs/safety_panel_gemma3_27b.jsonl

# (d) Interactive evidence-seeking (ED; one model only in the paper)
provena-med generate interactive \
    --n 300 --budget 3 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --out outputs/int_agent_llama31_8b.jsonl
```

To fan out the full 10-model x 5-cohort sweep on a SLURM cluster, use the panel
launchers in `scripts/`:

```bash
export PROVENA_ROOT=$PWD PROVENA_MODELS_DIR=$HOME/models
bash scripts/launch_staged_panel.sh 150 cardiac_mm
bash scripts/launch_dx_panel.sh     150
bash scripts/launch_safety_panel.sh
bash scripts/launch_mprobe_panel.sh 60 cardiac_mm     # M axis: target models
bash scripts/launch_int_panel.sh                       # interactive (Llama-3.1-8B only)
```

Each launcher submits one SLURM job per model and prints the job IDs.

## 5. Score (the axes)

Each axis has its own scorer. The held-out judge (Llama-3.3-70B by default) is
shared across the W and W x M axes; the safety axis is fully deterministic
(no model). All scorers print results and write JSONL.

```bash
# W axis: validity, judge-precision, recall, yield, guideline precision
provena-med eval w \
    --inputs outputs/staged_ed_gemma3_27b.jsonl \
    --judge  meta-llama/Llama-3.3-70B-Instruct

# Diagnosis Hit@1 / Hit@3 / Hit@5 (BioLORD soft match vs principal dx)
provena-med eval diagnosis \
    --glob 'outputs/dx_*.jsonl' \
    --out  outputs/dx_panel.jsonl

# Safety: source-traceable unsafe-prescribing rate
provena-med eval safety \
    --inputs outputs/safety_panel_gemma3_27b.jsonl

# Causal W (Shapley + necessity + sufficiency; cardiac, expensive)
provena-med eval causal \
    --inputs outputs/staged_cardiac_mm_gemma3_27b.jsonl --n 60

# M probe (attention knockout; per cited evidence unit, per claim)
provena-med eval m_probe \
    --inputs outputs/staged_cardiac_mm_gemma3_27b.jsonl --n 60 \
    --model  google/gemma-3-27b-it

# W x M panel (consumes the M-probe rows + judge; produces quadrants)
provena-med eval wxm \
    --cohort cardiac_mm \
    --ids gemma3_27b medgemma_27b llama31_8b ...
```

## 6. Aggregate to one leaderboard row per model

After all scorers have run, aggregate the task-specific JSONL outputs:

```bash
provena-med leaderboard --in outputs/ --out leaderboard.json
```

The exported JSON contains one row per model, with metrics averaged across all
available cohort-level scorer rows. It does not emit LaTeX.

## Smoke test (no PhysioNet, no GPU)

To verify the install before doing the full pipeline:

```bash
provena-med info     # should list verbs/nouns
python - <<'PY'
import json
from provena_med.core.mmais import extract_claims_lenient
for line in open("examples/synthetic_outputs.jsonl"):
    rec = json.loads(line)
    print(rec["stay_id"], len(extract_claims_lenient(rec["output"])), "claims")
PY
```

## Expected runtimes (single H200)

| step                                      | time           |
|-------------------------------------------|----------------|
| build all 5 cohorts                       | 20-40 min      |
| enrichers (labs/meds/demographics)        | 5-15 min       |
| CXR pixel findings (ICU; ~4k images)      | 1-2 h          |
| staged generation, 1 model x 1 cohort     | 5-20 min       |
| W judge scoring, 1 panel run              | 30-90 min      |
| M probe, 1 model x cardiac (n=60)         | 15-25 min      |
| full 10-model x 5-cohort panel            | ~24-36 h wall  |
