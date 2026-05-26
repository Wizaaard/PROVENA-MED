"""Interactive ED cohort for PROVENA-MED active-evidence-seeking track.

Initial state s_0 is information-poor (chief complaint + triage + demographics + vitals).
Revealable evidence the model may REQUEST: history (HPI), labs_and_tests, medications.
The full discharge note and the diagnosis field are NEVER revealed (answer leakage).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from provena_med.core.edcds import EDCDS_DIR, _parse_dx_field

REVEALABLE_KEYS = ["history", "labs_and_tests", "medications"]


def _read_clinical_data() -> pd.DataFrame:
    zp = EDCDS_DIR / "clinical_data.csv.zip"
    with zipfile.ZipFile(zp) as z:
        name = [
            n for n in z.namelist() if n.endswith("clinical_data.csv") and "MACOSX" not in n
        ][0]
        with z.open(name) as f:
            return pd.read_csv(
                f,
                usecols=[
                    "stay_id",
                    "HPI",
                    "tests",
                    "past_medication",
                    "primary_diagnosis",
                    "secondary_diagnosis",
                ],
            )


def load_interactive_cases(n: int | None = None, seed: int = 0) -> list[dict]:
    clin = _read_clinical_data()
    iai = pd.read_csv(
        EDCDS_DIR / "initial_assessment_info.csv",
        usecols=["stay_id", "triage", "chiefcomplaint"],
    )
    demo = pd.read_csv(EDCDS_DIR / "patient_demographics.csv")  # stay_id, patient_info
    vit = pd.read_csv(
        EDCDS_DIR / "vital_signs.csv", usecols=["stay_id", "initial_vitals"]
    )

    df = clin.merge(iai, on="stay_id").merge(demo, on="stay_id").merge(vit, on="stay_id")
    df = df.dropna(subset=["primary_diagnosis", "chiefcomplaint"])
    df = df[df["primary_diagnosis"].astype(str).str.len() > 2].reset_index(drop=True)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    cases = []
    for _, r in df.iterrows():
        s0 = (
            f"Chief complaint: {str(r['chiefcomplaint']).strip()}\n"
            f"Triage acuity (1=most urgent, 5=least): {str(r['triage']).strip()}\n"
            f"Demographics: {str(r['patient_info']).strip()}\n"
            f"Initial vitals: {str(r['initial_vitals']).strip()}"
        )
        revealable = {}
        for key, col in [
            ("history", "HPI"),
            ("labs_and_tests", "tests"),
            ("medications", "past_medication"),
        ]:
            val = str(r[col]).strip()
            if val and val.lower() != "nan":
                revealable[key] = val
        cases.append(
            {
                "stay_id": int(r["stay_id"]),
                "s0": s0,
                "revealable": revealable,
                "gold": {
                    "primary": _parse_dx_field(r["primary_diagnosis"]),
                    "secondary": _parse_dx_field(r["secondary_diagnosis"]),
                },
            }
        )
    return cases


if __name__ == "__main__":
    cases = load_interactive_cases(n=2, seed=0)
    print(f"loaded {len(cases)} interactive cases\n")
    for c in cases:
        print("=" * 70)
        print("stay_id:", c["stay_id"])
        print("--- s0 ---\n", c["s0"])
        print("--- revealable keys ---", list(c["revealable"].keys()))
        for k, v in c["revealable"].items():
            print(f"   [{k}] {v[:90]!r}")
        print("GOLD:", c["gold"])
