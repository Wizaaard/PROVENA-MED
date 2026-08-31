"""Loader for the MIMIC-IV-Ext-CDS emergency-department cohort (PROVENA-MED ED track).

Each case is an under-specified ED presentation (HPI + initial vitals + demographics)
with gold primary/secondary diagnoses. Text-only; no GPU needed.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from provena_med import DATA_ROOT

import pandas as pd

EDCDS_DIR = Path(
    str(DATA_ROOT / "Datasets/MIMIC-IV/mimic-iv-ext-cds/1.0.2")
)


def _parse_dx_field(value) -> list[str]:
    """Gold diagnosis fields may be a Python-list-like string, a delimited string,
    or a single label. Return a clean list of diagnosis strings."""
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return []
    # Try literal list first
    if s[0] in "[(":
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    # Fall back to splitting on common separators
    parts = re.split(r"[;\n]|,(?![^(]*\))", s)
    return [p.strip(" '\"") for p in parts if p.strip(" '\"")]


def load_cases(n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Load ED cases with a usable primary diagnosis. Optionally subsample n."""
    df = pd.read_csv(EDCDS_DIR / "diagnosis.csv")
    df = df.dropna(subset=["primary_diagnosis"])
    df = df[df["primary_diagnosis"].astype(str).str.len() > 2].reset_index(drop=True)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def case_presentation(row) -> str:
    """The under-specified initial presentation shown to the model."""
    return (
        f"Demographics: {str(row.get('patient_info', '')).strip()}\n"
        f"Initial vitals: {str(row.get('initial_vitals', '')).strip()}\n"
        f"History of present illness:\n{str(row.get('HPI', '')).strip()}"
    )


def gold_diagnoses(row) -> dict:
    primary = _parse_dx_field(row.get("primary_diagnosis"))
    secondary = _parse_dx_field(row.get("secondary_diagnosis"))
    return {"primary": primary, "secondary": secondary}


if __name__ == "__main__":
    df = load_cases(n=3, seed=0)
    print(f"loaded {len(df)} sample cases | columns: {list(df.columns)}\n")
    for _, row in df.iterrows():
        print("=" * 70)
        print("stay_id:", row["stay_id"])
        print(case_presentation(row)[:400])
        print("GOLD:", gold_diagnoses(row))
