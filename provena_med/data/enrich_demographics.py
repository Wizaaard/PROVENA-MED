"""Add explicit age/sex to every PROVENA-MED v0.1 cohort record (demographics metadata).

Needed for age-gated guideline rules (Beers >=65) and subgroup/fairness analysis. ED and
eICU already carry age in their text fields; ICU/cardiac/MIMIC-III are joined to the
patient tables (ICU & cardiac via MIMIC-IV hosp/patients; MIMIC-III via PATIENTS+ADMISSIONS,
with the >89 -> 90 age cap). Rewrites cohorts/*.jsonl in place with `age` and `sex_female`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from provena_med import DATA_ROOT

import pandas as pd

ROOT = DATA_ROOT / "PROVENA-MED/cohorts"
M4 = DATA_ROOT / "Datasets/MIMIC-IV/mimiciv/3.1"
CARD = DATA_ROOT / "Datasets/MIMIC-IV/mimic-iv-ext-cardiac-disease/1.0.0"
M3 = DATA_ROOT / "Datasets/MIMIC-III/mimiciii/1.4"


def _rewrite(cohort: str, age_of, sex_of):
    p = ROOT / f"{cohort}.jsonl"
    rows = [json.loads(l) for l in open(p)]
    hit = 0
    for c in rows:
        a, s = age_of(c), sex_of(c)
        if a is not None:
            c["age"] = a; hit += 1
        if s is not None:
            c["sex_female"] = s
    with p.open("w") as f:
        for c in rows:
            f.write(json.dumps(c) + "\n")
    print(f"  {cohort:11} {hit}/{len(rows)} got age")


def main():
    pat4 = pd.read_csv(M4 / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    age4 = dict(zip(pat4.subject_id, pat4.anchor_age))
    sex4 = dict(zip(pat4.subject_id, pat4.gender == "F"))

    # ED / eICU: already have age in text -> parse to explicit fields
    def parse_age(txt):
        m = re.search(r"age[:\s]+(\d{1,3})", str(txt).lower())
        return int(m.group(1)) if m else None

    def parse_sex(txt):
        t = str(txt).lower()
        return True if "female" in t else (False if "male" in t else None)

    _rewrite("ed", lambda c: parse_age(c.get("patient_info", "")),
             lambda c: parse_sex(c.get("patient_info", "")))
    _rewrite("eicu", lambda c: parse_age(c.get("demographics", "")),
             lambda c: parse_sex(c.get("demographics", "")))

    # ICU: subject_id -> MIMIC-IV patients
    _rewrite("icu_mm", lambda c: int(age4[c["subject_id"]]) if c.get("subject_id") in age4 else None,
             lambda c: bool(sex4[c["subject_id"]]) if c.get("subject_id") in sex4 else None)

    # cardiac: hadm_id -> subject_id (heart_diagnoses) -> MIMIC-IV patients
    hd = pd.read_csv(CARD / "heart_diagnoses.csv", usecols=["hadm_id", "subject_id"]).drop_duplicates("hadm_id")
    h2s = dict(zip(hd.hadm_id, hd.subject_id))
    _rewrite("cardiac_mm",
             lambda c: int(age4[h2s[c["id"]]]) if h2s.get(c["id"]) in age4 else None,
             lambda c: bool(sex4[h2s[c["id"]]]) if h2s.get(c["id"]) in sex4 else None)

    # MIMIC-III: HADM_ID -> SUBJECT_ID (ADMISSIONS) + ADMITTIME -> age via PATIENTS.DOB
    adm = pd.read_csv(M3 / "ADMISSIONS.csv.gz", usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"],
                      parse_dates=["ADMITTIME"])
    pat3 = pd.read_csv(M3 / "PATIENTS.csv.gz", usecols=["SUBJECT_ID", "GENDER", "DOB"], parse_dates=["DOB"])
    adm = adm.merge(pat3, on="SUBJECT_ID", how="left")
    age3, sex3 = {}, {}
    for _, r in adm.iterrows():
        if pd.isna(r.DOB) or pd.isna(r.ADMITTIME):
            continue
        # python datetime avoids pandas' ~292yr ns-Timedelta overflow for >89 (shifted DOB)
        yrs = (r.ADMITTIME.to_pydatetime() - r.DOB.to_pydatetime()).days / 365.25
        age3[int(r.HADM_ID)] = 90 if yrs > 89 else max(0, int(round(yrs)))
        sex3[int(r.HADM_ID)] = (r.GENDER == "F")
    _rewrite("mimic3", lambda c: age3.get(int(c["id"])), lambda c: sex3.get(int(c["id"])))


if __name__ == "__main__":
    main()
