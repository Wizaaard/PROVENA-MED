"""Recover FULL-STAY, TIMESTAMPED labs for all cohorts (extract once, window as a filter).

Rather than baking a 24h window into extraction, we pull every curated lab across the whole
admission with its time offset (hours from the stay anchor), downsample, and store `labs_ts`
= {labname: [[offset_h, value, flag], ...]}. The decision-time bundle then takes a window
VIEW (default [-12h,+24h]); a retrospective variant can take the full stay; the interactive
track reveals by time -- all without re-scanning. Anchors: cardiac=admittime,
icu=ICU intime, ED=ED arrival (via mimic-iv-ed edstays -> hadm), mimic3=ADMITTIME.

One awk pass over MIMIC-IV hosp/labevents covers cardiac+icu+ED; MIMIC-III is separate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.data.build_icu_cohort import LAB_ITEMS  # noqa: E402
from provena_med.core.bigtable import awk_filter, write_ids  # noqa: E402

PM = Path("<DATA_ROOT>/PROVENA-MED")
MC = Path("<DATA_ROOT>/Datasets/MIMIC-IV/mimiciv/3.1")
ED = Path("<DATA_ROOT>/Datasets/MIMIC-IV/mimic-iv-ed/2.2/ed")
M3 = Path("<DATA_ROOT>/Datasets/MIMIC-III/mimiciii/1.4")
M3_LAB = {v: k for k, v in LAB_ITEMS.items()}  # name->itemid (shared dictionary)


def downsample(points, bucket_h=12, max_n=8):
    """points = [(off_h, val, flag)] sorted; keep first per bucket, cap max_n."""
    points.sort(key=lambda p: p[0])
    out, seen = [], set()
    for off, val, flag in points:
        b = int(off // bucket_h)
        if b not in seen:
            seen.add(b)
            out.append([round(off, 1), round(float(val), 2), flag])
        if len(out) >= max_n:
            break
    return out


def _attach(cohort_file: Path, key_to_ts: dict, key="id"):
    rows = [json.loads(l) for l in open(cohort_file)]
    hit = 0
    for r in rows:
        ts = key_to_ts.get(int(r[key]))
        if ts:
            r["labs_ts"] = ts
            hit += 1
    with cohort_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return hit, len(rows)


def labs_ts_from_df(lv, anchor: dict, idcol: str):
    """lv has [idcol, itemid, charttime, valuenum, flag]; anchor maps id->Timestamp."""
    lv = lv.dropna(subset=["valuenum"]).copy()
    lv["t0"] = lv[idcol].map(anchor)
    lv = lv.dropna(subset=["t0", "charttime"])
    lv["off"] = (lv["charttime"] - lv["t0"]).dt.total_seconds() / 3600.0
    lv = lv[(lv.off >= -24) & (lv.off <= 24 * 30)]  # within a 30-day stay
    out: dict[int, dict] = {}
    for (k, itemid), g in lv.groupby([idcol, "itemid"]):
        name = LAB_ITEMS.get(int(itemid))
        if not name:
            continue
        pts = [(o, v, (str(fl) if pd.notna(fl) else "")) for o, v, fl in
               zip(g.off, g.valuenum, g.get("flag", pd.Series([None] * len(g))))]
        out.setdefault(int(k), {})[name] = downsample(pts)
    return out


def main():
    tmp = Path("outputs/tmp"); tmp.mkdir(parents=True, exist_ok=True)

    # ---- MIMIC-IV: gather anchors for cardiac + icu + ED, then ONE labevents scan ----
    adm = pd.read_csv(MC / "hosp/admissions.csv.gz", usecols=["hadm_id", "admittime"],
                      parse_dates=["admittime"])
    admit = dict(zip(adm.hadm_id.astype(int), adm.admittime))

    card = [json.loads(l) for l in open(PM / "cohorts" / "cardiac_mm.jsonl")]
    icu = [json.loads(l) for l in open(PM / "cohorts" / "icu_mm.jsonl")]
    ed = [json.loads(l) for l in open(PM / "cohorts" / "ed.jsonl")]

    # ED stay_id -> (hadm_id, ED intime)
    eds = pd.read_csv(ED / "edstays.csv.gz", usecols=["stay_id", "hadm_id", "intime"],
                      parse_dates=["intime"])
    ed_hadm = dict(zip(eds.stay_id.astype(int), eds.hadm_id))
    ed_intime = dict(zip(eds.stay_id.astype(int), eds.intime))

    hadm_anchor: dict[int, pd.Timestamp] = {}
    for c in card:
        if int(c["id"]) in admit:
            hadm_anchor[int(c["id"])] = admit[int(c["id"])]
    for c in icu:
        hadm_anchor[int(c["id"])] = pd.Timestamp(c["intime"])
    ed_stay_hadm = {}
    for c in ed:
        sid = int(c["id"]); h = ed_hadm.get(sid)
        if pd.notna(h):
            hadm_anchor[int(h)] = ed_intime.get(sid, admit.get(int(h)))
            ed_stay_hadm[sid] = int(h)
    hadms = set(hadm_anchor)
    print(f"[labs-ts] MIMIC-IV hadm anchors: {len(hadms)} (cardiac+icu+ED-admitted)")

    write_ids(hadms, tmp / "all_hadms.txt")
    print("[labs-ts] scanning hosp/labevents (full stay)...")
    lv = awk_filter(MC / "hosp/labevents.csv.gz", tmp / "all_hadms.txt", 3,
                    tmp / "all_lab.csv", item_field=5, items=LAB_ITEMS)
    lv["charttime"] = pd.to_datetime(lv["charttime"], errors="coerce")
    by_hadm = labs_ts_from_df(lv.rename(columns={"hadm_id": "hid"}), hadm_anchor, "hid")

    h, n = _attach(PM / "cohorts" / "cardiac_mm.jsonl", by_hadm); print(f"[labs-ts] cardiac {h}/{n}")
    h, n = _attach(PM / "cohorts" / "icu_mm.jsonl", by_hadm); print(f"[labs-ts] icu {h}/{n}")
    ed_ts = {sid: by_hadm[hh] for sid, hh in ed_stay_hadm.items() if hh in by_hadm}
    h, n = _attach(PM / "cohorts" / "ed.jsonl", ed_ts); print(f"[labs-ts] ED {h}/{n}")

    # ---- MIMIC-III: separate LABEVENTS scan (chunked, anchor=ADMITTIME) ----
    m3 = [json.loads(l) for l in open(PM / "cohorts" / "mimic3.jsonl")]
    m3_hadms = {int(r["id"]) for r in m3}
    adm3 = pd.read_csv(M3 / "ADMISSIONS.csv.gz", usecols=["HADM_ID", "ADMITTIME"], parse_dates=["ADMITTIME"])
    anchor3 = {int(h): t for h, t in zip(adm3.HADM_ID, adm3.ADMITTIME) if int(h) in m3_hadms}
    keep = set(LAB_ITEMS)
    print("[labs-ts] scanning MIMIC-III LABEVENTS...")
    parts = []
    for chunk in pd.read_csv(M3 / "LABEVENTS.csv.gz",
                             usecols=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "FLAG"],
                             chunksize=2_000_000, parse_dates=["CHARTTIME"]):
        chunk = chunk[chunk.HADM_ID.isin(m3_hadms) & chunk.ITEMID.isin(keep)]
        if len(chunk):
            parts.append(chunk)
    lv3 = pd.concat(parts) if parts else pd.DataFrame(columns=["HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "FLAG"])
    lv3 = lv3.rename(columns={"HADM_ID": "hid", "ITEMID": "itemid", "CHARTTIME": "charttime",
                              "VALUENUM": "valuenum", "FLAG": "flag"})
    by_h3 = labs_ts_from_df(lv3, anchor3, "hid")
    h, n = _attach(PM / "cohorts" / "mimic3.jsonl", by_h3); print(f"[labs-ts] mimic3 {h}/{n}")
    print("[labs-ts] done. labs_ts attached; window is now a filter (see provena.bundle.window_labs).")


if __name__ == "__main__":
    main()
