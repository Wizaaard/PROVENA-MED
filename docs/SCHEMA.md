# PROVENA-MED bundle schema

Every encounter is one JSON object on one line in a `cohorts/<cohort>.jsonl`.
The same shape is consumed by every task and scorer in the package.

## Top-level record

```jsonc
{
  "id":     94872,                       // int, stable within a cohort
  "header": "Cohort: ed. Chief complaint: ...; Patient: ...",
  "bundle": [ ... evidence-unit objects ... ],
  "gold":   { "primary": ["..."], "secondary": ["..."] },
  "flags":  { "leak_primary": false, "has_image": true, "triangulated": true, ... },

  // optional v0.2+ additions
  "reveal":          { "s0_visible": [...], "actions": { "take_history": [...], ... } },
  "guideline_rules": ["B3-FALLS", "FDA-OPIOID-BENZO"],
  "labs_ts":         { "creatinine": [[hours_from_admit, value, "high"], ...], ... },
  "home_meds":       ["lisinopril", "metformin"],
  "allergies":       ["penicillin"],
  "age": 71, "sex_female": 1, "conditions": ["heart_failure", "ckd_stage_3"],
  "triangulation_levels": { "primary_dx": 3, "ckd_stage_3": 2, ... }
}
```

## Evidence unit object (entries of `bundle`)

```jsonc
{
  "id":   "hpi:0",                       // unique within the bundle
  "type": "NOTE_SPAN",                   // see below
  "text": "...the literal text shown to the model..."
}
```

### Evidence types

| `type`            | id prefix                  | description                                              |
|-------------------|----------------------------|----------------------------------------------------------|
| `NOTE_SPAN`       | `hpi:`, `exam:`, `pmh:`    | sentence-level span from the HPI / exam / past history   |
| `TABLE_ROW`       | `vital:*`, `lab:*`         | one timestamped structured value                         |
| `IMAGE_FINDING`   | `cxr:*`, `ct:*`            | a radiology finding (report text or pixel-derived)       |
| `GUIDELINE_RULE`  | `B*`, `PH*`, `FDA-*`       | a sourced safety / DDI rule that **fires** for this case |

The model must cite by **exact** `id`; an id that does not appear in the
bundle counts against citation validity.

## Reasoning artifact (model output)

The staged-generation task asks the model for a JSON object of this shape:

```jsonc
{
  "problem_list": [{"claim": "...", "evidence": ["ID", ...]}],
  "differential": [{"diagnosis": "...", "rank": 1, "evidence": ["ID", ...]}],
  "workup":       [{"action": "...", "evidence": ["ID", ...]}],
  "therapy":      [{"action": "...", "evidence": ["ID", ...],
                    "safety_status": "allowed|caution|contraindicated|insufficient"}],
  "monitoring":   [{"action": "...", "evidence": ["ID", ...]}],
  "missing_information": ["..."]
}
```

Every claim must cite at least one evidence ID. Therapy claims may additionally
cite `GUIDELINE_RULE` IDs for the safety axis. The output is wrapped in a
` ```json ... ``` ` fence and parsed leniently (`provena_med.core.mmais.extract_claims_lenient`).

## Gold and triangulation

- `gold.primary` is a *list* in cohort-source order; `gold.primary[0]` is the
  principal admission diagnosis (the target for `Hit@k`). Other entries are
  comorbidities.
- `triangulation_levels[k]` is the corroboration count for label `k` across
  independent signals (billed code, narrative, structured lab signature, ...).
- `flags.triangulated` is true iff the primary label has corroboration level >= 2.
- `flags.leak_primary` marks the rare cases where the primary diagnosis appears
  verbatim in the narrative; report Hit@k on `flags.leak_primary == false` to be
  leak-free.

## File layout (after `provena-med build pack`)

```
data/PROVENA-MED-v0.3/
  cohorts/{ed,cardiac_mm,icu_mm,eicu,mimic3}.jsonl
  eval_split/
    cohorts/<cohort>.jsonl          # 2,400 per cohort
    splits.json                     # dev/test ids + sha256
    manifest.json
  guidelines/
    guideline_catalog.jsonl          # 49 citable GUIDELINE_RULE units
  pixel_findings/
    icu_pixel_findings.jsonl         # vision-model-extracted findings (ICU only)
  manifest.json                      # global counts + provenance
  README.md                          # dataset card
```

None of this is shipped in the released repo — `provena-med build pack`
materializes it from the user's own PhysioNet downloads.
