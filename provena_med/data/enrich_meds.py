"""Recover the home/pre-admission medication list + allergies for every cohort.

These are LEAKAGE-SAFE context (what the patient was on BEFORE this admission), unlike the
inpatient meds actually administered (which would reveal the treatment decision). Sources:
  ED        : mimic-iv-ed/medrecon (ED medication reconciliation) by stay_id
  ICU/cardiac: discharge-note "Medications on Admission:" + "Allergies:" sections (MIMIC-IV)
  MIMIC-III : same sections from NOTEEVENTS discharge summaries
  eICU      : admissionDrug + allergy tables
Writes `home_meds` (list) and `allergies` (text) into cohorts/*.jsonl (both v0.1 and v0.2),
so DDI and allergy guideline rules can fire and a med-reconciliation action appears.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.safety import _allergy, _med_tokens  # noqa: E402

PM = Path("<DATA_ROOT>/PROVENA-MED")
MIV = Path("<DATA_ROOT>/Datasets/MIMIC-IV")
ED = MIV / "mimic-iv-ed/2.2/ed"
NOTE = MIV / "mimic-iv-note/2.2/note/discharge.csv.gz"
M3 = Path("<DATA_ROOT>/Datasets/MIMIC-III/mimiciii/1.4")
E = Path("<DATA_ROOT>/Datasets/eICU/eicu-crd/2.0")

MED_SECTION = re.compile(r"Medications on Admission:?\s*(.*?)(?:\n\s*\n|Discharge Medications|"
                         r"Discharge Disposition|Discharge Condition|Facility:)", re.S | re.I)


def meds_from_note(text: str) -> list[str]:
    m = MED_SECTION.search(str(text))
    if not m:
        return []
    out = []
    for line in _med_tokens(m.group(1)):
        # drop dosage/route tail; keep the leading drug name tokens
        name = re.split(r"\s+\d|\s+\(|,| - | tab| cap| mg| mcg| unit| ml| puff| po | iv ",
                        line, flags=re.I)[0].strip()
        if 2 < len(name) < 40 and not name.lower().startswith(("none", "see ", "as ")):
            out.append(name)
    return out[:25]


def _save(cohort: str, meds: dict, allg: dict, key="id"):
    for sub in ["cohorts", "v0.2/cohorts"]:
        fp = PM / sub / f"{cohort}.jsonl"
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in open(fp)]
        for r in rows:
            k = int(r[key])
            if k in meds:
                r["home_meds"] = meds[k]
            if k in allg:
                r["allergies"] = allg[k]
        with fp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    nm = sum(1 for v in meds.values() if v)
    print(f"  {cohort:11} home_meds {nm} | allergies {sum(1 for v in allg.values() if v)}")


def hadm_set(cohort):
    return {int(r["id"]) for r in (json.loads(l) for l in open(PM / "cohorts" / f"{cohort}.jsonl"))}


def main():
    # ---- ED: medrecon by stay_id ----
    mr = pd.read_csv(ED / "medrecon.csv.gz", usecols=["stay_id", "name"])
    ed_ids = hadm_set("ed")
    mr = mr[mr.stay_id.isin(ed_ids)]
    ed_meds = mr.groupby("stay_id")["name"].apply(
        lambda s: sorted({str(x).strip() for x in s if str(x).strip()})[:25]).to_dict()
    _save("ed", {int(k): v for k, v in ed_meds.items()}, {})

    # ---- MIMIC-IV ICU + cardiac: discharge-note sections ----
    want = hadm_set("icu_mm") | hadm_set("cardiac_mm")
    meds, allg = {}, {}
    for chunk in pd.read_csv(NOTE, usecols=["hadm_id", "text"], chunksize=40000):
        chunk = chunk[chunk.hadm_id.isin(want)]
        for _, r in chunk.iterrows():
            h = int(r.hadm_id)
            if h in meds:
                continue
            meds[h] = meds_from_note(r.text)
            atext, _ = _allergy(str(r.text))
            allg[h] = atext
    _save("icu_mm", meds, allg)
    _save("cardiac_mm", meds, allg)

    # ---- MIMIC-III: NOTEEVENTS discharge summaries ----
    m3_ids = hadm_set("mimic3")
    meds3, allg3 = {}, {}
    for chunk in pd.read_csv(M3 / "NOTEEVENTS.csv.gz", usecols=["HADM_ID", "CATEGORY", "TEXT"],
                             chunksize=50000, dtype={"HADM_ID": "float64"}):
        chunk = chunk[chunk.HADM_ID.isin(m3_ids) & chunk.CATEGORY.str.strip().str.lower().str.startswith("discharge")]
        for _, r in chunk.iterrows():
            h = int(r.HADM_ID)
            if h in meds3:
                continue
            meds3[h] = meds_from_note(r.TEXT)
            atext, _ = _allergy(str(r.TEXT))
            allg3[h] = atext
    _save("mimic3", meds3, allg3)

    # ---- eICU: admissionDrug + allergy ----
    try:
        ad = pd.read_csv(E / "admissionDrug.csv.gz", usecols=["patientunitstayid", "drugname"])
        eids = hadm_set("eicu")
        ad = ad[ad.patientunitstayid.isin(eids)]
        emeds = ad.groupby("patientunitstayid")["drugname"].apply(
            lambda s: sorted({str(x).strip().lower() for x in s if str(x).strip()})[:25]).to_dict()
        al = pd.read_csv(E / "allergy.csv.gz", usecols=["patientunitstayid", "allergyname"])
        al = al[al.patientunitstayid.isin(eids)]
        eallg = al.groupby("patientunitstayid")["allergyname"].apply(
            lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()}))).to_dict()
        _save("eicu", {int(k): v for k, v in emeds.items()}, {int(k): v for k, v in eallg.items()})
    except Exception as e:
        print("  eicu meds/allergy skipped:", type(e).__name__, str(e)[:80])


if __name__ == "__main__":
    main()
