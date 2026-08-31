"""Materialize the multimodal ICU cohort (one-time heavy extraction -> JSONL cache).

Anchors on MIMIC-IV icu/icustays, restricted to stays whose subject has a chest
radiograph AND whose admission has a discharge note. For each stay we assemble the
admission-time evidence (windowed to intime + WINDOW_H hours to avoid outcome leakage):
  NOTE_SPAN     <- HPI section of the discharge note
  TABLE_ROW     <- first-24h labs (curated itemids) + ICU vitals (chartevents)
  IMAGE_FINDING <- a real CXR DICOM for the subject (pixel findings added later)
Gold = billed ICD diagnoses (hosp/diagnoses_icd x d_icd_diagnoses) for the admission.

The chartevents (~430M rows) and labevents (~150M rows) tables are pre-filtered with awk
(itemid + id-set) before pandas, so the scan is a single streaming pass each.

  python build_icu_cohort.py --n-pool 4000 --seed 0 --out outputs/icu_mm_cohort.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from provena_med import DATA_ROOT

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.cxr_image import dicom_full_path  # noqa: E402

M = DATA_ROOT / "Datasets/MIMIC-IV"
MC = M / "mimiciv/3.1"
NOTE = M / "mimic-iv-note/2.2/note/discharge.csv.gz"
CXR_REC = M / "mimic-cxr/2.1.0/cxr-record-list.csv.gz"
WINDOW_H = 24

VITAL_ITEMS = {220045: "heart_rate", 220050: "sbp", 220179: "sbp", 220051: "dbp",
               220180: "dbp", 220052: "mbp", 220181: "mbp", 220210: "resp_rate",
               220277: "o2_sat", 223762: "temp_c", 223761: "temp_f"}
LAB_ITEMS = {50912: "creatinine", 50971: "potassium", 50983: "sodium", 50902: "chloride",
             50882: "bicarbonate", 51006: "bun", 50931: "glucose", 50960: "magnesium",
             50970: "phosphate", 51221: "hematocrit", 51222: "hemoglobin", 51301: "wbc",
             51265: "platelets", 50813: "lactate", 50820: "ph", 50818: "pco2",
             50821: "po2", 51237: "inr", 51275: "ptt", 50861: "alt", 50878: "ast",
             50885: "bilirubin_total", 50862: "albumin", 51003: "troponin_t",
             50963: "ntprobnp"}


def _section(text: str, start: str, stops: list[str]) -> str:
    stop = "|".join(stops + [r"\n\s*\n"])
    m = re.search(rf"{start}\s*:(.*?)(?:{stop})", str(text), re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def awk_filter(table: Path, idfile: Path, id_field: int, item_field: int,
               itemids, header: list[str], out: Path) -> pd.DataFrame:
    """Stream `table` keeping rows whose id_field is in idfile and item_field matches itemids."""
    items_re = "^(" + "|".join(str(i) for i in itemids) + ")$"
    prog = (f'NR==FNR{{s[$1]=1; next}} FNR==1{{print; next}} '
            f'(${id_field} in s) && (${item_field} ~ items)')
    with out.open("wb") as fout:
        p1 = subprocess.Popen(["zcat", str(table)], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["awk", "-F,", "-v", f"items={items_re}", prog,
                               str(idfile), "-"], stdin=p1.stdout, stdout=fout)
        p1.stdout.close()
        p2.communicate()
    return pd.read_csv(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/icu_mm_cohort.jsonl")
    args = ap.parse_args()

    tmp = Path("outputs/tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    # ---- 1. candidate pool: ICU stays with a CXR (subject) and a discharge note (hadm) ----
    icu = pd.read_csv(MC / "icu/icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id", "intime"],
                      parse_dates=["intime"])
    cxr_subj = set(pd.read_csv(CXR_REC, usecols=["subject_id"])["subject_id"].unique())
    dn_hadm = set(pd.read_csv(NOTE, usecols=["hadm_id"])["hadm_id"].dropna().astype(int))
    pool = icu[icu.subject_id.isin(cxr_subj) & icu.hadm_id.isin(dn_hadm)]
    pool = pool.sort_values("stay_id").drop_duplicates("hadm_id")  # one stay per admission
    pool = pool.sample(n=min(args.n_pool, len(pool)), random_state=args.seed).reset_index(drop=True)
    print(f"[icu] candidate pool {len(pool)} stays")
    hadms = set(pool.hadm_id.astype(int)); stays = set(pool.stay_id.astype(int))
    intime = dict(zip(pool.stay_id.astype(int), pool.intime))
    intime_h = dict(zip(pool.hadm_id.astype(int), pool.intime))
    pd.Series(sorted(stays)).to_csv(tmp / "stayids.txt", index=False, header=False)
    pd.Series(sorted(hadms)).to_csv(tmp / "hadmids.txt", index=False, header=False)

    # ---- 2. gold ICD diagnoses ----
    dx = pd.read_csv(MC / "hosp/diagnoses_icd.csv.gz")
    dx = dx[dx.hadm_id.isin(hadms)]
    dd = pd.read_csv(MC / "hosp/d_icd_diagnoses.csv.gz")
    dx = dx.merge(dd, on=["icd_code", "icd_version"], how="left").sort_values(["hadm_id", "seq_num"])
    gold = dx.groupby("hadm_id")["long_title"].apply(lambda s: [x for x in s if isinstance(x, str)]).to_dict()
    print(f"[icu] gold for {len(gold)} admissions")

    # ---- 3. discharge-note sections ----
    notes = {}
    for chunk in pd.read_csv(NOTE, usecols=["hadm_id", "text"], chunksize=50000):
        chunk = chunk[chunk.hadm_id.isin(hadms)]
        for _, r in chunk.iterrows():
            h = int(r.hadm_id)
            if h in notes:
                continue
            t = r.text
            notes[h] = {
                "chief_complaint": _section(t, "Chief Complaint", ["Major Surg", "History of Present"]),
                "HPI": _section(t, "History of Present Illness",
                                ["Past Medical History", "Social History", "REVIEW OF SYSTEMS", "Brief Hospital"]),
                "physical_exam": _section(t, "Physical Exam", ["Pertinent Results", "Brief Hospital", "Discharge"]),
            }
    print(f"[icu] notes parsed for {len(notes)} admissions")

    # ---- 4. CXR DICOM per subject (first study) ----
    rec = pd.read_csv(CXR_REC, usecols=["subject_id", "path"])
    rec = rec[rec.subject_id.isin(set(pool.subject_id))]
    first_cxr = rec.groupby("subject_id")["path"].first().to_dict()

    # ---- 5. vitals (chartevents) windowed to first WINDOW_H h ----
    print("[icu] scanning chartevents (vitals)...")
    cv = awk_filter(MC / "icu/chartevents.csv.gz", tmp / "stayids.txt", 3, 7,
                    VITAL_ITEMS, [], tmp / "cv.csv")
    cv["charttime"] = pd.to_datetime(cv["charttime"], errors="coerce")
    cv["t0"] = cv["stay_id"].map(intime)
    cv = cv[(cv.charttime >= cv.t0) & (cv.charttime <= cv.t0 + pd.Timedelta(hours=WINDOW_H))]
    cv["vital"] = cv["itemid"].map(VITAL_ITEMS)
    cv = cv.sort_values("charttime").dropna(subset=["valuenum"])
    vitals = {}
    for (sid, vit), g in cv.groupby(["stay_id", "vital"]):
        vitals.setdefault(int(sid), {})[vit] = round(float(g.valuenum.iloc[0]), 1)
    print(f"[icu] vitals for {len(vitals)} stays")

    # ---- 6. labs (labevents) windowed ----
    print("[icu] scanning labevents (labs)...")
    lv = awk_filter(MC / "hosp/labevents.csv.gz", tmp / "hadmids.txt", 3, 5,
                    LAB_ITEMS, [], tmp / "lv.csv")
    lv["charttime"] = pd.to_datetime(lv["charttime"], errors="coerce")
    lv["t0"] = lv["hadm_id"].map(intime_h)
    lv = lv[(lv.charttime >= lv.t0) & (lv.charttime <= lv.t0 + pd.Timedelta(hours=WINDOW_H))]
    lv["lab"] = lv["itemid"].map(LAB_ITEMS)
    lv = lv.sort_values("charttime").dropna(subset=["valuenum"])
    labs = {}
    for (h, lab), g in lv.groupby(["hadm_id", "lab"]):
        row = g.iloc[0]
        labs.setdefault(int(h), {})[lab] = {"value": round(float(row.valuenum), 2),
                                            "uom": str(row.get("valueuom", "")),
                                            "flag": str(row.get("flag", "")) if pd.notna(row.get("flag", "")) else ""}
    print(f"[icu] labs for {len(labs)} admissions")

    # ---- 7. assemble + write ----
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for _, r in pool.iterrows():
            h, sid, subj = int(r.hadm_id), int(r.stay_id), int(r.subject_id)
            nt = notes.get(h, {})
            if not nt.get("HPI") or not gold.get(h):
                continue  # need at least HPI + gold
            f.write(json.dumps({
                "id": h, "stay_id": sid, "subject_id": subj,
                "intime": str(r.intime),
                "chief_complaint": nt.get("chief_complaint", ""),
                "HPI": nt.get("HPI", ""), "physical_exam": nt.get("physical_exam", ""),
                "vitals": vitals.get(sid, {}), "labs": labs.get(h, {}),
                "dicom_path": dicom_full_path(first_cxr[subj]) if subj in first_cxr else "",
                "gold": gold.get(h, []),
            }) + "\n")
            n += 1
    print(f"[icu] wrote {n} complete ICU multimodal cases -> {out}")


if __name__ == "__main__":
    main()
