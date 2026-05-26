"""Multimodal ICU cohort loader (reads the materialized cache from build_icu_cohort.py).

Each case = an ICU admission with admission-time (first-24h) evidence: HPI (NOTE_SPAN),
labs + vitals (TABLE_ROW), and a real chest-radiograph DICOM (IMAGE_FINDING via pixels).
Gold = billed ICD diagnoses for the admission.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "outputs" / "icu_mm_cohort.jsonl"


def load_icu_cases(n: int | None = None, seed: int = 0, cache: str | None = None) -> list[dict]:
    path = Path(cache) if cache else CACHE
    cases = [json.loads(line) for line in open(path)]
    if n is not None and n < len(cases):
        idx = sorted(random.Random(seed).sample(range(len(cases)), n))
        cases = [cases[i] for i in idx]
    return cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from provena_med.core.bundle import build_bundle_icu
    cs = load_icu_cases(3, seed=0)
    print(f"loaded {len(cs)} ICU multimodal cases")
    for c in cs:
        b = build_bundle_icu(c, [])
        nmod: dict = {}
        for u in b:
            nmod[u["type"]] = nmod.get(u["type"], 0) + 1
        print(f"\nhadm {c['id']} stay {c['stay_id']} | units {nmod} | "
              f"vitals {list(c['vitals'])} | labs {list(c['labs'])[:6]} | gold {c['gold'][:1]}")
        print("  dicom:", c["dicom_path"].split("/files/")[-1] if c["dicom_path"] else "(none)")
