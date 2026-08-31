"""Augment PROVENA-MED v0.1 -> v0.2: add the interactive reveal schema, GUIDELINE_RULE
evidence (catalog + per-case applicable rules), and merged triangulation levels.

Reads cohorts/*.jsonl (v0.1), and for every case:
  * rebuilds the typed evidence bundle with the cohort's builder,
  * derives the patient's conditions / age / eGFR (where available) and attaches the
    APPLICABLE GUIDELINE_RULE units (so therapy claims can cite the guideline that fires),
  * computes the interactive reveal schema (s0 + requestable actions) from that bundle,
  * merges triangulation corroboration levels where a manifest exists.
Writes v0.2/cohorts/*.jsonl + guidelines/guideline_catalog.jsonl + manifest/README.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from provena_med import DATA_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core import normalize, safety  # noqa: E402
from provena_med.core.bundle import (build_bundle, build_bundle_eicu, build_bundle_icu,  # noqa: E402
                            build_bundle_mimic3, build_bundle_mm, window_labs)
from provena_med.core.guidelines import load_catalog, relevant_units  # noqa: E402
from provena_med.core.reveal import build_reveal  # noqa: E402

ROOT = DATA_ROOT / "PROVENA-MED"
CODE_OUT = Path(__file__).resolve().parent / "outputs"

BUILDERS = {
    "ed": lambda c: build_bundle(c),
    "cardiac_mm": build_bundle_mm,
    "icu_mm": lambda c: build_bundle_icu(c),  # uses case['image_findings'] (radiologist report)
    "eicu": build_bundle_eicu,
    "mimic3": build_bundle_mimic3,
}


def _creatinine_of(c: dict):
    """Creatinine from the windowed labs (labs_ts/labs) or ED 'tests' text."""
    labs = window_labs(c)
    rec = labs.get("creatinine") if isinstance(labs, dict) else None
    v = rec.get("value") if isinstance(rec, dict) else rec
    try:
        return float(v)
    except (TypeError, ValueError):
        return safety._creatinine(c.get("tests", "") or "")


def context(cohort: str, c: dict):
    """Per-case context for guideline relevance: conditions, age, eGFR, current meds, allergy."""
    texts = [c.get("HPI", ""), c.get("physical_exam", "")]
    if cohort == "eicu":
        texts += [" ".join(c.get("past_history", []))]
    texts += [" ".join(map(str, c.get("gold", [])))]
    conditions = safety._conditions(*texts)
    age = c.get("age")
    female = c.get("sex_female")
    if age is None:
        age, female = safety._demo(c.get("patient_info", "") or c.get("demographics", ""))
    egfr = None
    scr = _creatinine_of(c)
    if scr and age is not None and female is not None:
        egfr = safety.ckd_epi_egfr(scr, age, female)
    current_classes = {cl for m in (c.get("home_meds") or []) for cl in normalize.drug_classes(m)}
    allergy = bool(c.get("allergies"))
    return conditions, age, egfr, current_classes, allergy


def load_triangulation():
    out = {}
    for ch in BUILDERS:
        p = CODE_OUT / f"triang_{ch}.jsonl"
        if p.exists():
            d = {}
            for line in open(p):
                r = json.loads(line)
                d[int(r["id"])] = {lab["label"]: lab["level"] for lab in r["labels"]}
            out[ch] = d
    return out


def main():
    v2 = ROOT / "v0.2"
    (v2 / "cohorts").mkdir(parents=True, exist_ok=True)
    (ROOT / "guidelines").mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    with (ROOT / "guidelines" / "guideline_catalog.jsonl").open("w") as f:
        for u in catalog:
            f.write(json.dumps(u) + "\n")
    print(f"guideline catalog: {len(catalog)} GUIDELINE_RULE units")

    triang = load_triangulation()
    manifest = {"name": "PROVENA-MED", "version": "0.3", "cohorts": {}, "guideline_units": len(catalog),
                "notes": "v0.3: timestamped full-stay labs (windowed), home-meds + allergies, DDI activation"}
    total = 0
    for cohort, builder in BUILDERS.items():
        src = ROOT / "cohorts" / f"{cohort}.jsonl"
        if not src.exists():
            print(f"  {cohort}: v0.1 missing, skip"); continue
        rows = [json.loads(l) for l in open(src)]
        tri = triang.get(cohort, {})
        n_guid = n_tri = 0
        with (v2 / "cohorts" / f"{cohort}.jsonl").open("w") as f:
            for c in rows:
                conditions, age, egfr, cur_classes, allergy = context(cohort, c)
                grules = relevant_units(conditions, egfr, age, allergy=allergy,
                                        current_classes=cur_classes)
                bundle = builder(c) + grules
                reveal = build_reveal(bundle)  # order_imaging now lists real report IMAGE_FINDING IDs
                c["pixels_available"] = bool(c.get("dicom_path"))  # VLMs can load dicom_path
                c["conditions"] = sorted(conditions)
                c["guideline_rules"] = [u["id"] for u in grules]
                c["reveal"] = reveal
                lvl = tri.get(int(c["id"]))
                if lvl:
                    c["triangulation_levels"] = lvl
                    c["gold_triangulated"] = [g for g, v in lvl.items() if v >= 2]
                    n_tri += 1
                n_guid += int(bool(grules))
                f.write(json.dumps(c) + "\n")
        manifest["cohorts"][cohort] = {"n": len(rows), "with_applicable_guideline": n_guid,
                                       "with_triangulation": n_tri}
        total += len(rows)
        print(f"  {cohort:11} {len(rows):6} cases | applicable-guideline {n_guid} | triangulated {n_tri}")
    manifest["total_encounters"] = total
    with (v2 / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPROVENA-MED v0.2 -> {v2}  ({total} encounters)")


if __name__ == "__main__":
    main()
