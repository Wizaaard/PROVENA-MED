"""Recover structured first-24h labs for the cardiac cohort by joining back to MIMIC-IV.

The cardiac cohort (mimic-iv-ext-cardiac-disease) was derived from MIMIC-IV, so its labs
live upstream in hosp/labevents -- we just hadn't pulled them into the bundle (only notes +
imaging). Each cardiac record's id is a real MIMIC-IV hadm_id, so we awk-prefilter
labevents to those admissions + curated itemids, window to admittime+24h, and write a
structured `labs` dict back into the cohort files. This adds a TABLE_ROW modality to
cardiac AND lets eGFR/renal guideline rules fire.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from provena_med import DATA_ROOT

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.data.build_icu_cohort import LAB_ITEMS  # noqa: E402  (shared curated itemids)
from provena_med.core.bigtable import awk_filter, write_ids  # noqa: E402

MC = DATA_ROOT / "Datasets/MIMIC-IV/mimiciv/3.1"
PM = DATA_ROOT / "PROVENA-MED"
WINDOW_H = 24


def main():
    files = [PM / "cohorts" / "cardiac_mm.jsonl", PM / "v0.2" / "cohorts" / "cardiac_mm.jsonl"]
    rows = [json.loads(l) for l in open(files[0])]
    hadms = {int(r["id"]) for r in rows}
    print(f"[cardiac-labs] {len(hadms)} admissions")

    adm = pd.read_csv(MC / "hosp/admissions.csv.gz", usecols=["hadm_id", "admittime"],
                      parse_dates=["admittime"])
    admit = dict(zip(adm.hadm_id.astype(int), adm.admittime))
    print(f"[cardiac-labs] admittime for {sum(h in admit for h in hadms)}/{len(hadms)} admissions")

    tmp = Path("outputs/tmp"); tmp.mkdir(parents=True, exist_ok=True)
    write_ids(hadms, tmp / "cardiac_hadms.txt")
    print("[cardiac-labs] scanning labevents...")
    lv = awk_filter(MC / "hosp/labevents.csv.gz", tmp / "cardiac_hadms.txt", 3,
                    tmp / "cardiac_lab.csv", item_field=5, items=LAB_ITEMS)
    lv["charttime"] = pd.to_datetime(lv["charttime"], errors="coerce")
    lv["t0"] = lv["hadm_id"].map(admit)
    lv = lv[(lv.charttime >= lv.t0) & (lv.charttime <= lv.t0 + pd.Timedelta(hours=WINDOW_H))]
    lv = lv.sort_values("charttime").dropna(subset=["valuenum"])
    labs: dict[int, dict] = {}
    for (h, itemid), g in lv.groupby(["hadm_id", "itemid"]):
        name = LAB_ITEMS.get(int(itemid))
        if not name:
            continue
        row = g.iloc[0]
        labs.setdefault(int(h), {})[name] = {
            "value": round(float(row.valuenum), 2),
            "uom": str(row.get("valueuom", "")) if pd.notna(row.get("valueuom", "")) else "",
            "flag": str(row.get("flag", "")) if pd.notna(row.get("flag", "")) else ""}
    print(f"[cardiac-labs] recovered labs for {len(labs)}/{len(hadms)} admissions "
          f"(mean {sum(len(v) for v in labs.values())/max(1,len(labs)):.1f} labs/case)")

    for fp in files:
        rs = [json.loads(l) for l in open(fp)]
        for r in rs:
            r["labs"] = labs.get(int(r["id"]), {})
        with fp.open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        print(f"[cardiac-labs] updated {fp}")


if __name__ == "__main__":
    main()
