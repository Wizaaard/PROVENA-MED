"""Consolidate all cohorts into the PROVENA-MED dataset directory.

Materializes every cohort to a uniform JSONL (cohort, id, source, gold, gold_primary,
leak_primary, + the evidence fields needed to rebuild bundles), copies the pixel-finding
caches, and writes a manifest + data card. `leak_primary` flags cases whose PRIMARY gold
diagnosis appears verbatim in the note text (so the diagnosis metric can be reported
leak-free); comorbidity overlap is intentionally not flagged.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from provena_med import DATA_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import (build_bundle, build_bundle_eicu, build_bundle_icu,  # noqa: E402
                            build_bundle_mimic3, build_bundle_mm)

ROOT = DATA_ROOT / "PROVENA-MED"
CODE_OUT = Path(__file__).resolve().parent / "outputs"


def note_text(bundle) -> str:
    return " ".join(u["text"] for u in bundle if u["type"] == "NOTE_SPAN").lower()


def leak_primary(primary: str, bundle) -> bool:
    p = str(primary).lower().strip()
    return len(p) >= 6 and p in note_text(bundle)


def materialize():
    from provena_med.core.edcds import load_cases, gold_diagnoses
    from provena_med.core.cohort_mm import load_mm_cases
    from provena_med.core.cohort_icu import load_icu_cases
    from provena_med.core.cohort_eicu import load_eicu_cases
    from provena_med.core.cohort_mimic3 import load_mimic3_cases

    out: dict[str, list[dict]] = {}

    # ED
    rows = []
    for _, r in load_cases(None, 0).iterrows():
        c = r.to_dict()
        g = gold_diagnoses(c)
        gold = g["primary"] + g["secondary"]
        prim = (g["primary"] or [""])[0]
        rows.append({"cohort": "ed", "id": int(c["stay_id"]), "source": "mimic-iv-ext-cds",
                     "gold": gold, "gold_primary": prim,
                     "leak_primary": leak_primary(prim, build_bundle(c)),
                     "HPI": c.get("HPI", ""), "initial_vitals": c.get("initial_vitals", ""),
                     "tests": c.get("tests", ""), "patient_info": c.get("patient_info", "")})
    out["ed"] = rows

    # cardiac multimodal (report-text imaging)
    rows = []
    for c in load_mm_cases(None, 0):
        prim = (c["gold"] or [""])[0]
        rows.append({"cohort": "cardiac_mm", "id": int(c["id"]), "source": "mimic-iv-ext-cardiac-disease",
                     "gold": c["gold"], "gold_primary": prim,
                     "leak_primary": leak_primary(prim, build_bundle_mm(c)),
                     "HPI": c.get("HPI", ""), "physical_exam": c.get("physical_exam", ""),
                     "chief_complaint": c.get("chief_complaint", ""),
                     **{k: c.get(k, "") for k in ["X-ray", "CT", "Ultrasound", "MRI", "ECG", "CATH"]}})
    out["cardiac_mm"] = rows

    # ICU multimodal (true-pixel imaging)
    rows = []
    for c in load_icu_cases(None, 0):
        prim = (c["gold"] or [""])[0]
        rec = dict(c); rec.update({"cohort": "icu_mm", "source": "mimic-iv-icu+cxr",
                                   "gold_primary": prim,
                                   "leak_primary": leak_primary(prim, build_bundle_icu(c, []))})
        rows.append(rec)
    out["icu_mm"] = rows

    # eICU external (structured)
    rows = []
    for c in load_eicu_cases(None, 0):
        prim = (c["gold"] or [""])[0]
        rec = dict(c); rec.update({"cohort": "eicu", "gold_primary": prim,
                                   "leak_primary": leak_primary(prim, build_bundle_eicu(c))})
        rows.append(rec)
    out["eicu"] = rows

    # MIMIC-III external (text + report imaging)
    rows = []
    for c in load_mimic3_cases(None, 0):
        prim = (c["gold"] or [""])[0]
        rec = dict(c); rec.update({"cohort": "mimic3", "gold_primary": prim,
                                   "leak_primary": leak_primary(prim, build_bundle_mimic3(c))})
        rows.append(rec)
    out["mimic3"] = rows
    return out


def main():
    (ROOT / "cohorts").mkdir(parents=True, exist_ok=True)
    (ROOT / "pixel_findings").mkdir(parents=True, exist_ok=True)
    data = materialize()

    manifest = {"name": "PROVENA-MED", "version": "0.1", "cohorts": {}}
    total = 0
    for name, rows in data.items():
        path = ROOT / "cohorts" / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        leak = sum(r["leak_primary"] for r in rows)
        manifest["cohorts"][name] = {"n": len(rows), "source": rows[0]["source"],
                                     "primary_leak": leak, "primary_leak_pct": round(100 * leak / len(rows), 1)}
        total += len(rows)
        print(f"  {name:11} {len(rows):6} cases  (primary-leak {leak}, {100*leak/len(rows):.0f}%) -> {path}")
    manifest["total_encounters"] = total

    for src in ["icu_pixel_findings.jsonl", "cxr_pixel_findings.jsonl"]:
        s = CODE_OUT / src
        if s.exists():
            shutil.copy(s, ROOT / "pixel_findings" / src)
    with (ROOT / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPROVENA-MED v0.1: {total} encounters across {len(data)} cohorts -> {ROOT}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
