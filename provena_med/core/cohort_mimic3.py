"""MIMIC-III external cohort loader (reads the cache from build_mimic3_cohort.py)."""
from __future__ import annotations

import json
import random
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "outputs" / "mimic3_cohort.jsonl"


def load_mimic3_cases(n: int | None = None, seed: int = 0, cache: str | None = None) -> list[dict]:
    cases = [json.loads(line) for line in open(Path(cache) if cache else CACHE)]
    if n is not None and n < len(cases):
        idx = sorted(random.Random(seed).sample(range(len(cases)), n))
        cases = [cases[i] for i in idx]
    return cases


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from collections import Counter
    from provena_med.core.bundle import build_bundle_mimic3
    for c in load_mimic3_cases(3, 0):
        b = build_bundle_mimic3(c)
        print(f"\nhadm {c['id']} | units {Counter(u['type'] for u in b)} | gold {c['gold'][:2]}")
        for u in b[:5]:
            print("   ", u["id"], "::", u["text"][:60])
