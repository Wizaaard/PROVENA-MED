"""Materialize ground-truth IMAGE_FINDING evidence for the ICU cohort from the RADIOLOGIST
report (not a model), keeping the DICOM pixels for VLM evaluation.

MIMIC-CXR ships each study's radiologist report as a .txt co-located with the study dir
(.../sXXXX.txt, sibling of the .../sXXXX/ image dir). For every ICU case we read that
report, extract the FINDINGS/IMPRESSION sentences as `image_findings` (human ground truth),
and keep `dicom_path` so a vision-language model can be evaluated against this gold. This
replaces using CheXagent's reading as evidence (which would bake one model's perception into
the benchmark); CheXagent/LLaVA-Med are evaluated SYSTEMS, not the evidence source.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import extract_image_findings  # noqa: E402

PM = Path("<DATA_ROOT>/PROVENA-MED")


def report_path(dicom_path: str) -> str:
    # .../files/pX/pXXXX/sYYYY/<dicom>.dcm -> .../files/pX/pXXXX/sYYYY.txt
    return os.path.dirname(dicom_path) + ".txt"


def main():
    files = [PM / "cohorts" / "icu_mm.jsonl", PM / "v0.2" / "cohorts" / "icu_mm.jsonl"]
    base = [json.loads(l) for l in open(files[0])]
    findings_by_id, n_units, n_rep = {}, 0, 0
    for c in base:
        dp = c.get("dicom_path", "")
        rp = report_path(dp) if dp else ""
        fnd = []
        if rp and os.path.exists(rp):
            n_rep += 1
            try:
                txt = open(rp, errors="ignore").read()
            except OSError:
                txt = ""
            fnd = [u["text"] for u in extract_image_findings(txt)][:12]
        findings_by_id[int(c["id"])] = fnd
        n_units += len(fnd)
    print(f"[icu-cxr-report] reports found {n_rep}/{len(base)} | "
          f"{n_units} radiologist IMAGE_FINDING units ({n_units/max(1,n_rep):.1f}/case)")

    for fp in files:
        rows = [json.loads(l) for l in open(fp)]
        for c in rows:
            c["image_findings"] = findings_by_id.get(int(c["id"]), [])
            c["cxr_report_available"] = bool(c["image_findings"])
        with fp.open("w") as f:
            for c in rows:
                f.write(json.dumps(c) + "\n")
        print(f"[icu-cxr-report] updated {fp}")


if __name__ == "__main__":
    main()
