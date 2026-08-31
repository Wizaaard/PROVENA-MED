"""Freeze the official PROVENA-MED evaluation split (~12k encounters).

Balanced 2,400 encounters per cohort (= 12,000), split dev(200)/test(2,200), sampled
deterministically (seed 0). Each record carries slicing flags so metrics can be reported on
the full set and on sub-slices (leak-free diagnosis, guideline-applicable therapy,
triangulated gold, has-pixels multimodal). The split is content-hashed (sha256 of the
sorted id list per cohort) so it is reproducible and citable. Eval-only: dev is for
prompt/threshold calibration, test for reporting.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from provena_med import DATA_ROOT

PM = DATA_ROOT / "PROVENA-MED"
SRC = PM / "v0.2" / "cohorts"
OUT = PM / "eval_split"
COHORTS = ["ed", "cardiac_mm", "icu_mm", "eicu", "mimic3"]
PER_COHORT, N_DEV, SEED = 2400, 200, 0


_CARDIAC_IMG = ["X-ray", "CT", "Ultrasound", "MRI", "ECG", "CATH"]


def _has_image(c: dict) -> bool:
    if c.get("image_findings"):
        return True
    # cardiac stores imaging in modality columns consumed by build_bundle_mm
    return any(str(c.get(k, "")).strip() not in ("", "nan", "[]") for k in _CARDIAC_IMG)


def flags(c: dict) -> dict:
    return {
        "leak_primary": bool(c.get("leak_primary")),
        "has_guideline": bool(c.get("guideline_rules")),
        "has_image": _has_image(c),
        "has_pixels": bool(c.get("pixels_available") or c.get("dicom_path")),
        "has_labs": bool(c.get("labs_ts") or c.get("labs")),
        "has_meds": bool(c.get("home_meds")),
        "triangulated": bool(c.get("gold_triangulated")),
        "n_gold": len(c.get("gold", [])),
    }


def main():
    (OUT / "cohorts").mkdir(parents=True, exist_ok=True)
    splits, manifest = {}, {"name": "PROVENA-MED", "split": "eval-v1", "seed": SEED,
                            "per_cohort": PER_COHORT, "dev_per_cohort": N_DEV, "cohorts": {}}
    grand = 0
    for ch in COHORTS:
        rows = [json.loads(l) for l in open(SRC / f"{ch}.jsonl")]
        idx = list(range(len(rows)))
        random.Random(SEED).shuffle(idx)
        pick = idx[:min(PER_COHORT, len(rows))]
        dev = set(pick[:N_DEV])
        sel = []
        for i in pick:
            c = rows[i]
            c["split"] = "dev" if i in dev else "test"
            c["flags"] = flags(c)
            sel.append(c)
        with (OUT / "cohorts" / f"{ch}.jsonl").open("w") as f:
            for c in sel:
                f.write(json.dumps(c) + "\n")
        ids = sorted(int(c["id"]) for c in sel)
        h = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]
        splits[ch] = {"dev": sorted(int(c["id"]) for c in sel if c["split"] == "dev"),
                      "test": sorted(int(c["id"]) for c in sel if c["split"] == "test"),
                      "sha256_16": h}
        agg = {k: sum(c["flags"][k] for c in sel) for k in
               ("leak_primary", "has_guideline", "has_image", "has_pixels", "has_labs",
                "has_meds", "triangulated")}
        manifest["cohorts"][ch] = {"n": len(sel), "dev": len(dev), "test": len(sel) - len(dev),
                                   "sha256_16": h, **agg}
        grand += len(sel)
        print(f"  {ch:11} {len(sel):5} (dev {len(dev)}/test {len(sel)-len(dev)}) "
              f"img {agg['has_image']} pix {agg['has_pixels']} guide {agg['has_guideline']} "
              f"leak {agg['leak_primary']} hash {h}")
    manifest["total"] = grand
    with (OUT / "splits.json").open("w") as f:
        json.dump(splits, f, indent=2)
    with (OUT / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nPROVENA-MED eval split: {grand} encounters (dev {len(COHORTS)*N_DEV} / "
          f"test {grand - len(COHORTS)*N_DEV}) -> {OUT}")


if __name__ == "__main__":
    main()
