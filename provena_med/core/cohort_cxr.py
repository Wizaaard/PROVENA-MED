"""Multimodal image cohort: cardiac cases linked to a real MIMIC-CXR radiograph.

Each case carries the patient's HPI + report-derived finding text (the citable
IMAGE_FINDING units) AND the path to an actual chest-radiograph DICOM, so a
vision-language model sees the pixels. (Temporal alignment of study<->admission is a
known caveat: we take a CXR for the subject; admission-window matching is future work.)
"""
from __future__ import annotations

from pathlib import Path

from provena_med import DATA_ROOT

import pandas as pd

from provena_med.core.cxr_image import dicom_full_path

CARDIAC = DATA_ROOT / "Datasets/MIMIC-IV/mimic-iv-ext-cardiac-disease/1.0.0"
CXR = DATA_ROOT / "Datasets/MIMIC-IV/mimic-cxr/2.1.0"
IMG_COLS = ["X-ray", "CT", "Ultrasound", "MRI", "ECG", "CATH"]


def load_cxr_cases(n: int | None = None, seed: int = 0) -> list[dict]:
    df = pd.read_csv(CARDIAC / "heart_diagnoses.csv")
    gold = (pd.read_csv(CARDIAC / "heart_diagnoses_all_true.csv")
            .groupby("hadm_id")["long_title"].apply(list).to_dict())
    # one CXR DICOM per subject (first frontal-ish study)
    rec = pd.read_csv(CXR / "cxr-record-list.csv.gz", usecols=["subject_id", "path"])
    first_cxr = rec.groupby("subject_id")["path"].first().to_dict()

    df = df[df["HPI"].astype(str).str.len() > 20]
    df = df[df["subject_id"].isin(first_cxr)].reset_index(drop=True)
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    cases = []
    for _, r in df.iterrows():
        subj = int(r["subject_id"])
        case = {
            "id": int(r["hadm_id"]), "subject_id": subj,
            "gold": gold.get(int(r["hadm_id"]), []),
            "HPI": str(r.get("HPI", "")), "physical_exam": str(r.get("physical_exam", "")),
            "chief_complaint": str(r.get("chief_complaint", "")).strip(),
            "dicom_path": dicom_full_path(first_cxr[subj]),
        }
        for c in IMG_COLS:
            case[c] = r.get(c, "")
        cases.append(case)
    return cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from provena_med.core.bundle import build_bundle_mm
    from provena_med.core.cxr_image import load_dicom_image
    cases = load_cxr_cases(3, seed=0)
    print(f"loaded {len(cases)} cardiac+CXR cases")
    for c in cases:
        b = build_bundle_mm(c)
        nmod = {}
        for u in b:
            nmod[u["type"]] = nmod.get(u["type"], 0) + 1
        img = load_dicom_image(c["dicom_path"])
        print(f"\nhadm {c['id']} subj {c['subject_id']} | units {nmod} | image {img.size} | gold {c['gold'][:1]}")
        print("  dicom:", c["dicom_path"].split("/files/")[-1])
