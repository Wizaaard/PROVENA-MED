"""Materialize the eICU-CRD external cohort (one-time extraction -> JSONL cache).

eICU is the STRUCTURED external arm: its notes are templated UI breadcrumbs (no narrative),
so evidence comes from structured sources:
  NOTE_SPAN  <- past medical history + physical-exam findings (short structured units)
  TABLE_ROW  <- first-24h labs (curated) + periodic vitals
  (no imaging)
Gold = the leaf terms of billed diagnosisstrings for the unit stay. Offsets are minutes
relative to unit admission; we window evidence to [-12h, +24h].

  python build_eicu_cohort.py --n-pool 3000 --seed 0 --out outputs/eicu_cohort.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bigtable import awk_filter, write_ids  # noqa: E402

E = Path("<DATA_ROOT>/Datasets/eICU/eicu-crd/2.0")
WIN_LO, WIN_HI = -720, 1440  # minutes

ELAB = {"creatinine": "creatinine", "potassium": "potassium", "sodium": "sodium",
        "chloride": "chloride", "bicarbonate": "bicarbonate", "HCO3": "bicarbonate",
        "BUN": "bun", "glucose": "glucose", "Hgb": "hemoglobin", "Hct": "hematocrit",
        "WBC x 1000": "wbc", "platelets x 1000": "platelets", "lactate": "lactate",
        "pH": "ph", "paCO2": "pco2", "paO2": "po2", "PT - INR": "inr", "PTT": "ptt",
        "ALT (SGPT)": "alt", "AST (SGOT)": "ast", "total bilirubin": "bilirubin_total",
        "albumin": "albumin", "magnesium": "magnesium", "calcium": "calcium",
        "anion gap": "anion_gap", "troponin - I": "troponin", "troponin - T": "troponin"}
EVITAL = {"heartrate": "heart_rate", "sao2": "o2_sat", "respiration": "resp_rate",
          "temperature": "temp_c", "systemicsystolic": "sbp",
          "systemicdiastolic": "dbp", "systemicmean": "mbp"}
QUALIFIERS = {"known", "suspected", "s/p"}


def dx_leaf(s: str) -> str:
    parts = [p.strip() for p in str(s).split("|") if p.strip()]
    while parts and parts[-1].lower() in QUALIFIERS:
        parts.pop()
    return parts[-1] if parts else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/eicu_cohort.jsonl")
    args = ap.parse_args()
    tmp = Path("outputs/tmp"); tmp.mkdir(parents=True, exist_ok=True)

    # ---- 1. pool of unit stays ----
    pat = pd.read_csv(E / "patient.csv.gz",
                      usecols=["patientunitstayid", "age", "gender", "unittype",
                               "hospitaladmitsource"])
    pat = pat.sample(n=min(args.n_pool, len(pat)), random_state=args.seed).reset_index(drop=True)
    stays = set(pat.patientunitstayid.astype(int))
    write_ids(stays, tmp / "eicu_stays.txt")
    print(f"[eicu] pool {len(pat)} unit stays")

    # ---- 2. gold diagnosis leaves ----
    dx = pd.read_csv(E / "diagnosis.csv.gz",
                     usecols=["patientunitstayid", "diagnosisstring", "diagnosisoffset"])
    dx = dx[dx.patientunitstayid.isin(stays)]
    gold: dict[int, list[str]] = {}
    for sid, g in dx.groupby("patientunitstayid"):
        leaves = []
        for s in g.sort_values("diagnosisoffset").diagnosisstring:
            leaf = dx_leaf(s)
            if leaf and leaf not in leaves:
                leaves.append(leaf)
        gold[int(sid)] = leaves
    print(f"[eicu] gold for {len(gold)} stays")

    # ---- 3. past history + physical exam (NOTE_SPAN sources) ----
    ph = awk_filter(E / "pastHistory.csv.gz", tmp / "eicu_stays.txt", 2, tmp / "eicu_ph.csv")
    ph = ph[ph.patientunitstayid.isin(stays)]
    past: dict[int, list[str]] = {}
    for sid, g in ph.groupby("patientunitstayid"):
        vals = [str(v).strip() for v in g["pasthistoryvaluetext"].dropna().unique()
                if str(v).strip() and str(v).strip().lower() != "no health problems"]
        if vals:
            past[int(sid)] = vals[:15]
    pe = awk_filter(E / "physicalExam.csv.gz", tmp / "eicu_stays.txt", 2, tmp / "eicu_pe.csv")
    pe = pe[pe.patientunitstayid.isin(stays)]
    exam: dict[int, list[str]] = {}
    for sid, g in pe.groupby("patientunitstayid"):
        vals = [str(v).strip() for v in g["physicalexamtext"].dropna().unique() if str(v).strip()]
        if vals:
            exam[int(sid)] = vals[:20]
    print(f"[eicu] past-history for {len(past)} | physical-exam for {len(exam)} stays")

    # ---- 4. labs (curated, windowed) ----
    lab = awk_filter(E / "lab.csv.gz", tmp / "eicu_stays.txt", 2, tmp / "eicu_lab.csv")
    lab = lab[lab.patientunitstayid.isin(stays) & lab.labname.isin(ELAB)
              & lab.labresultoffset.between(WIN_LO, WIN_HI)]
    lab = lab.sort_values("labresultoffset")
    labs: dict[int, dict] = {}
    for (sid, name), g in lab.groupby(["patientunitstayid", "labname"]):
        v = g.labresult.iloc[0]
        if pd.notna(v):
            labs.setdefault(int(sid), {})[ELAB[name]] = {
                "value": round(float(v), 2) if str(v).replace(".", "").replace("-", "").isdigit() else v,
                "uom": str(g.get("labmeasurenamesystem", pd.Series([""])).iloc[0]), "flag": ""}
    print(f"[eicu] labs for {len(labs)} stays")

    # ---- 5. periodic vitals (windowed, first reading) ----
    vp = awk_filter(E / "vitalPeriodic.csv.gz", tmp / "eicu_stays.txt", 2, tmp / "eicu_vp.csv")
    vp = vp[vp.patientunitstayid.isin(stays) & vp.observationoffset.between(0, WIN_HI)]
    vp = vp.sort_values("observationoffset")
    vitals: dict[int, dict] = {}
    for sid, g in vp.groupby("patientunitstayid"):
        d = {}
        for col, name in EVITAL.items():
            s = g[col].dropna()
            if len(s):
                d[name] = round(float(s.iloc[0]), 1)
        if d:
            vitals[int(sid)] = d
    print(f"[eicu] vitals for {len(vitals)} stays")

    # ---- 6. assemble ----
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for _, r in pat.iterrows():
            sid = int(r.patientunitstayid)
            if not gold.get(sid) or (sid not in labs and sid not in vitals):
                continue
            f.write(json.dumps({
                "id": sid, "source": "eicu",
                "demographics": f"age {r.age}, {r.gender}, {r.unittype}",
                "past_history": past.get(sid, []), "physical_exam": exam.get(sid, []),
                "vitals": vitals.get(sid, {}), "labs": labs.get(sid, {}),
                "gold": gold.get(sid, []),
            }) + "\n")
            n += 1
    print(f"[eicu] wrote {n} external eICU cases -> {out}")


if __name__ == "__main__":
    main()
