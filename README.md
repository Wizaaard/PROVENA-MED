---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - medical
  - clinical
  - mimic
  - eicu
  - benchmark
  - provenance
  - faithfulness
  - multimodal
  - evaluation-only
pretty_name: PROVENA-MED
size_categories:
  - 10K<n<100K
configs:
  - config_name: ed
    description: Emergency-department CDS cohort (primary; interactive home).
  - config_name: cardiac_mm
    description: MIMIC-IV-Ext-Cardiac (notes + recovered labs + radiologist report imaging).
  - config_name: icu_mm
    description: MIMIC-IV ICU x MIMIC-CXR (notes + labs + real radiograph pixels).
  - config_name: eicu
    description: eICU-CRD external (structured; different health system).
  - config_name: mimic3
    description: MIMIC-III external (ICD-9 era; older platform).
---

# PROVENA-MED

A clinician-free, **evaluation-only** benchmark for **provenance-faithful**,
multimodal clinical reasoning over linked EHR data, clinical notes, and chest
radiographs. Models produce a staged clinical artifact (problem list -> differential
-> workup -> therapy -> monitoring) in which every claim must cite typed evidence
units. Scoring is fully automatic: we verify whether each citation is *valid* (the
cited ID exists in the bundle), *supported* (judged by a held-out LLM), and
*causally used* by the model (via attention knockout on open-weight systems),
alongside diagnostic accuracy, drug safety, and interactive evidence-seeking
efficiency.

> **No new clinician annotation is used.** Gold is triangulated from pre-existing
> billed codes, structured clinical facts, deterministic guideline rules, and
> counterfactual provenance tests, so the benchmark scales with cohorts without
> annotation cost.

---

## Important: this repository ships *code and schema*, not patient data

PROVENA-MED is derived from **credentialed PhysioNet** resources (MIMIC-IV,
MIMIC-IV-ED-CDS, MIMIC-IV-Ext-Cardiac, MIMIC-CXR, MIMIC-III, eICU-CRD). PhysioNet's
data use agreement **forbids** redistribution of patient bundles. To use this
benchmark you must:

1. Obtain PhysioNet credentialed access to each underlying dataset.
2. Clone this repository and `pip install -e .` (or `pip install provena-med`).
3. Set `PROVENA_DATA_ROOT` to the directory containing `Datasets/` and
   `PROVENA-MED/`, then run the local build pipeline against your PhysioNet data.
4. Run the evaluation harness against your models of choice.

Everything in steps 2-4 is deterministic and reproducible from the released code.
The frozen 12,000-encounter evaluation split is content-hashed (see
`provena_med/data/eval_split.py`), so different users build byte-identical bundles.

## Dataset availability

The Hugging Face dataset page is [provena-med/provena-med](https://huggingface.co/datasets/provena-med/provena-med).
It is currently a metadata and release placeholder: no patient-level data, cohort bundles,
or experiment outputs are distributed there. A public data release will be considered only
after the source-data licenses and redistribution rights have been confirmed.

---

## Cohorts

| cohort       |       n | source                          | role                                       | modalities                                  |
|--------------|--------:|---------------------------------|--------------------------------------------|---------------------------------------------|
| `ed`         |   9,137 | MIMIC-IV-Ext-CDS (ED)           | primary internal; interactive track home   | NOTE_SPAN + TABLE_ROW                       |
| `cardiac_mm` |   4,761 | MIMIC-IV-Ext-Cardiac            | multimodal (report-text imaging)           | NOTE_SPAN + IMAGE_FINDING                   |
| `icu_mm`     |   3,963 | MIMIC-IV ICU x MIMIC-CXR        | multimodal ICU; **true CXR pixels**        | NOTE_SPAN + TABLE_ROW + IMAGE_FINDING       |
| `eicu`       |   2,585 | eICU-CRD 2.0                    | external (different health system)         | NOTE_SPAN + TABLE_ROW                       |
| `mimic3`     |   3,000 | MIMIC-III 1.4                   | external (ICD-9 era)                       | NOTE_SPAN + TABLE_ROW + IMAGE_FINDING       |
| **total**    | **23,446** |                              |                                            | + a 4th `GUIDELINE_RULE` modality across cohorts |

A **frozen 12,000-encounter evaluation split** (2,400/cohort; dev 200 + test 2,200)
is defined deterministically and content-hashed in `eval_split.json`. All numbers
in the paper use this split.

---

## Evidence units (typed, stable IDs)

Each case is serialized as a JSON bundle of evidence units that the model must
cite *by exact ID*:

- `NOTE_SPAN:*` - a sentence span of the HPI / past history / physical exam.
- `TABLE_ROW:vital:*`, `TABLE_ROW:lab:*` - a timestamped structured value.
- `IMAGE_FINDING:cxr:*` - in `icu_mm`, machine-extracted from the **actual chest
  radiograph pixels** (vision model); in `cardiac_mm` / `mimic3`, radiologist
  report findings.
- `GUIDELINE_RULE:*` - one of 49 source-traceable rules from the Phansalkar 2012
  high-priority DDI list, the AGS Beers Criteria 2023, and FDA Structured Product
  Labels. Surfaced *only* when the case actually triggers the rule (condition
  present, renal threshold crossed, age >= 65, drug class match).

---

## What this benchmark measures

The released task-specific scorers report the following metrics per model:

| axis                      | metric(s)                                            | what it asks                                                                 |
|---------------------------|------------------------------------------------------|------------------------------------------------------------------------------|
| Diagnosis                 | Hit@1, Hit@3 (BioLORD soft match)                    | Did the model name the principal admission diagnosis in its top-k?           |
| Attribution to world (W)  | Validity, Precision, Recall, **Yield**, Soundness, Guide-P | Are citations real IDs, and does the cited evidence support the claim?       |
| W x M faithfulness        | True (W+M+), Misgrounded (W-M+)                      | Does the model's *computation* actually route through the citations it makes? |
| Safety                    | Unsafe-recommendation rate                           | Does the proposed medication list violate authoritative drug-safety rules?   |

**M is computed by attention knockout** (masking the cited unit's key positions
while length-preserving), so it transfers across open-weight Transformer
architectures without per-model surgery. Closed-weight models are evaluable on
the **W axis only**.

An **interactive evidence-seeking** track measures whether the model gathers
information *efficiently* under a request budget, rather than just consuming a
fully-revealed record (`provena-med eval interactive`).

---

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Point the package at your credentialed local data.
export PROVENA_DATA_ROOT=/path/to/data-root

# 3. Build one cohort (requires PhysioNet data paths)
provena-med build ed_cardiac \
  --mimic-iv-ed-cds /path/to/mimic-iv-ext-cds \
  --mimic-iv-cardiac /path/to/mimic-iv-ext-cardiac \
  --out ./data

# 3. Run a model end-to-end on a small slice
provena-med generate staged \
  --provena \
  --cohort ed \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --n 30 \
  --out ./outputs/staged_ed_llama31_8b.jsonl

# 4. Score it (W axis, with the held-out LLM judge)
provena-med eval w \
  --in ./outputs/staged_ed_llama31_8b.jsonl

# 5. Aggregate released scorer JSONL outputs once all task-specific scorers have run.
provena-med leaderboard --in ./outputs --out leaderboard.json
```

Full build + 10-model panel reproduction commands live in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

---

## Repository layout

```
provena-med/
  README.md                    # this dataset card
  LICENSE                      # MIT for the code; data licensed by PhysioNet
  pyproject.toml
  provena_med/
    data/                      # cohort builders (read your PhysioNet copy)
    tasks/                     # staged-gen, dx, safety, interactive prompts
    eval/                      # W judge, M probe, safety, Hit@k scorers
    judges/                    # held-out LLM judge wrapper
    baselines/                 # the 10-model panel as a runnable harness
    cli.py                     # `provena-med` entrypoint
  scripts/                     # SLURM templates + one-shot launchers
  examples/                    # synthetic 5-case fixtures for smoke tests
  docs/                        # protocol + reproducibility checklist + leaderboard format
```

---

## Citation

If you use PROVENA-MED, please cite:

```bibtex
@misc{provenamed2026,
  title  = {{PROVENA-MED}: Benchmarking Interactive, Provenance-Faithful Multimodal Clinical Reasoning},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review.}
}
```

*Authors and affiliation are omitted under the venue's double-blind policy.*

---

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Data is **not** included; the underlying clinical resources are credentialed and
remain under the PhysioNet data use agreement.
