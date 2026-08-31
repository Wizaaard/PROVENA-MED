"""Multimodal cohort loader (cardiac extension) for PROVENA-MED.

Each case has HPI + physical exam (text) and X-ray/CT/Ultrasound/MRI/ECG finding text
(imaging), with ICD diagnoses as gold by hadm_id. Yields cases for build_bundle_mm.
"""
from __future__ import annotations

from pathlib import Path

from provena_med import DATA_ROOT

import pandas as pd

CARDIAC = Path(
    str(DATA_ROOT / "Datasets/MIMIC-IV/mimic-iv-ext-cardiac-disease/1.0.0")
)
IMG_COLS = ["X-ray", "CT", "Ultrasound", "MRI", "ECG", "CATH"]


def _has_imaging(row) -> bool:
    return any(len(str(row[c]).strip()) > 5 and str(row[c]).lower() != "nan"
               for c in ["X-ray", "CT", "Ultrasound", "MRI"])


def load_mm_cases(n: int | None = None, seed: int = 0) -> list[dict]:
    df = pd.read_csv(CARDIAC / "heart_diagnoses.csv")
    gold = (pd.read_csv(CARDIAC / "heart_diagnoses_all_true.csv")
            .groupby("hadm_id")["long_title"].apply(list).to_dict())
    df = df[df["HPI"].astype(str).str.len() > 20]
    df = df[df.apply(_has_imaging, axis=1)].reset_index(drop=True)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    cases = []
    for _, r in df.iterrows():
        hadm = int(r["hadm_id"])
        case = {
            "id": hadm,
            "gold": gold.get(hadm, []),
            "HPI": str(r.get("HPI", "")),
            "physical_exam": str(r.get("physical_exam", "")),
            "chief_complaint": str(r.get("chief_complaint", "")).strip(),
        }
        for c in IMG_COLS:
            case[c] = r.get(c, "")
        cases.append(case)
    return cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from provena_med.core.bundle import build_bundle_mm
    cases = load_mm_cases(3, seed=0)
    print(f"loaded {len(cases)} multimodal cases")
    for c in cases:
        b = build_bundle_mm(c)
        mods = {}
        for u in b:
            mods[u["type"]] = mods.get(u["type"], 0) + 1
        print(f"\nhadm {c['id']} | units={len(b)} {mods} | gold={c['gold'][:2]}")
        for u in [u for u in b if u["type"] == "IMAGE_FINDING"][:4]:
            print(f"   [{u['id']}] {u['text'][:80]}")
