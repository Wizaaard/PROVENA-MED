"""Materialize the MIMIC-III external cohort (one-time extraction -> JSONL cache).

MIMIC-III is the TEXT + report-imaging external arm (an older hospital era, ICD-9 coding):
  NOTE_SPAN     <- HPI section of the discharge summary (NOTEEVENTS)
  TABLE_ROW     <- first-24h labs (curated itemids, shared with MIMIC-IV)
  IMAGE_FINDING <- FINDINGS/IMPRESSION sentences of radiology reports (report text;
                   MIMIC-III has no DICOM pixels)
Gold = billed ICD-9 diagnoses (DIAGNOSES_ICD x D_ICD_DIAGNOSES) for the admission.

NOTEEVENTS has embedded commas/newlines so everything is read with pandas (chunked),
never awk.

  python build_mimic3_cohort.py --n-pool 3000 --seed 0 --out outputs/mimic3_cohort.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import extract_image_findings  # noqa: E402

T = Path("<DATA_ROOT>/Datasets/MIMIC-III/mimiciii/1.4")
WINDOW_H = 24
LAB_ITEMS = {50912: "creatinine", 50971: "potassium", 50983: "sodium", 50902: "chloride",
             50882: "bicarbonate", 51006: "bun", 50931: "glucose", 50960: "magnesium",
             50970: "phosphate", 51221: "hematocrit", 51222: "hemoglobin", 51301: "wbc",
             51265: "platelets", 50813: "lactate", 50820: "ph", 50818: "pco2",
             50821: "po2", 51237: "inr", 51275: "ptt", 50861: "alt", 50878: "ast",
             50885: "bilirubin_total", 50862: "albumin", 51003: "troponin_t", 50963: "ntprobnp"}


def _section(text: str, start: str, stops: list[str]) -> str:
    stop = "|".join(stops + [r"\n\s*\n"])
    m = re.search(rf"{start}\s*:(.*?)(?:{stop})", str(text), re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/mimic3_cohort.jsonl")
    args = ap.parse_args()

    # ---- 1. pool: ICU admissions with a discharge summary ----
    icu = pd.read_csv(T / "ICUSTAYS.csv.gz", usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"],
                      parse_dates=["INTIME"]).sort_values("ICUSTAY_ID").drop_duplicates("HADM_ID")
    icu = icu.sample(n=min(args.n_pool * 2, len(icu)), random_state=args.seed)  # oversample; filtered later
    hadms = set(icu.HADM_ID.astype(int)); intime_h = dict(zip(icu.HADM_ID.astype(int), icu.INTIME))
    print(f"[m3] candidate pool {len(icu)} ICU admissions")

    # ---- 2. gold ICD-9 ----
    dx = pd.read_csv(T / "DIAGNOSES_ICD.csv.gz")
    dx = dx[dx.HADM_ID.isin(hadms)]
    dd = pd.read_csv(T / "D_ICD_DIAGNOSES.csv.gz")
    dx = dx.merge(dd, on="ICD9_CODE", how="left").sort_values(["HADM_ID", "SEQ_NUM"])
    gold = dx.groupby("HADM_ID")["LONG_TITLE"].apply(
        lambda s: [x for x in s if isinstance(x, str)]).to_dict()
    print(f"[m3] gold for {len(gold)} admissions")

    # ---- 3. notes: discharge HPI + radiology findings (chunked, multiline-safe) ----
    hpi: dict[int, dict] = {}
    rad: dict[int, str] = {}
    for chunk in pd.read_csv(T / "NOTEEVENTS.csv.gz",
                             usecols=["HADM_ID", "CATEGORY", "TEXT"], chunksize=50000,
                             dtype={"HADM_ID": "float64"}):
        chunk = chunk[chunk.HADM_ID.isin(hadms)]
        for _, r in chunk.iterrows():
            h = int(r.HADM_ID)
            cat = str(r.CATEGORY).strip().lower()
            if cat.startswith("discharge") and h not in hpi:
                t = r.TEXT
                hpi[h] = {
                    "chief_complaint": _section(t, "Chief Complaint", ["Major Surg", "History of Present"]),
                    "HPI": _section(t, "History of Present Illness",
                                    ["Past Medical History", "Social History", "Brief Hospital", "REVIEW OF SYSTEMS"]),
                    "physical_exam": _section(t, "Physical Exam", ["Pertinent Results", "Brief Hospital"]),
                }
            elif cat.startswith("radiology") and h not in rad:
                rad[h] = str(r.TEXT)
    print(f"[m3] discharge HPI for {len(hpi)} | radiology for {len(rad)} admissions")

    # ---- 4. labs (curated, windowed; chunked) ----
    labs: dict[int, dict] = {}
    keep = set(LAB_ITEMS)
    for chunk in pd.read_csv(T / "LABEVENTS.csv.gz",
                             usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM", "FLAG"],
                             chunksize=1_000_000, parse_dates=["CHARTTIME"]):
        chunk = chunk[chunk.HADM_ID.isin(hadms) & chunk.ITEMID.isin(keep)]
        for _, r in chunk.iterrows():
            h = int(r.HADM_ID)
            t0 = intime_h.get(h)
            if t0 is None or pd.isna(r.CHARTTIME) or pd.isna(r.VALUENUM):
                continue
            if not (t0 <= r.CHARTTIME <= t0 + pd.Timedelta(hours=WINDOW_H)):
                continue
            name = LAB_ITEMS[int(r.ITEMID)]
            d = labs.setdefault(h, {})
            if name not in d:  # first in window
                d[name] = {"value": round(float(r.VALUENUM), 2),
                           "uom": str(r.VALUEUOM) if pd.notna(r.VALUEUOM) else "",
                           "flag": str(r.FLAG) if pd.notna(r.FLAG) else ""}
    print(f"[m3] labs for {len(labs)} admissions")

    # ---- 5. assemble ----
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for h in icu.HADM_ID.astype(int):
            nt = hpi.get(h)
            if not nt or not nt.get("HPI") or not gold.get(h):
                continue
            img = [u["text"] for u in extract_image_findings(rad.get(h, ""))][:12]
            f.write(json.dumps({
                "id": h, "source": "mimic3",
                "chief_complaint": nt.get("chief_complaint", ""),
                "HPI": nt.get("HPI", ""), "physical_exam": nt.get("physical_exam", ""),
                "labs": labs.get(h, {}), "image_findings": img,
                "gold": gold.get(h, []),
            }) + "\n")
            n += 1
            if n >= args.n_pool:
                break
    print(f"[m3] wrote {n} external MIMIC-III cases -> {out}")


if __name__ == "__main__":
    main()
