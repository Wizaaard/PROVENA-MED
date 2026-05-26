"""Derive drug-safety rules from official FDA labels (openFDA) — source-traceable.

For each drug we pull its FDA label and extract structured rules ONLY where the label
states them: renal-function thresholds, pregnancy contraindications, and the presence of
a boxed warning. Every rule records the FDA set_id + retrieval date so it is verifiable.
No rule logic is authored from memory; thresholds are read from the label text.

Usage:
  HF_HUB_OFFLINE=1 python build_rules_fda.py --out provena/rules/drug_safety_fda.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.normalize import drug_classes  # noqa: E402
from provena_med.core.sources import fda_label  # noqa: E402

# common drugs spanning our rule classes (extend freely; each is verified against FDA)
SEED_DRUGS = [
    "metformin", "ibuprofen", "naproxen", "ketorolac", "celecoxib", "indomethacin",
    "lisinopril", "enalapril", "ramipril", "losartan", "spironolactone",
    "warfarin", "apixaban", "rivaroxaban", "oxycodone", "morphine", "hydromorphone",
    "lorazepam", "alprazolam", "diazepam", "nitrofurantoin", "atorvastatin",
    "simvastatin", "methotrexate", "sumatriptan", "metoclopramide", "gabapentin",
    "fluoxetine", "sertraline", "azithromycin", "fluconazole",
]

RENAL_PATTERNS = [
    r"egfr\s*(?:below|<|less than|under|of less than)\s*(\d{2})",
    r"creatinine clearance\s*(?:below|<|less than|under|of less than)\s*(\d{2})",
    r"\bcrcl\s*(?:below|<|less than|under)\s*(\d{2})",
]


def _snippet(text: str, span_center: int, width: int = 120) -> str:
    a = max(0, span_center - width // 2)
    return re.sub(r"\s+", " ", text[a:a + width]).strip()


def extract_rules(drug: str) -> list[dict]:
    lab = fda_label(drug)
    if not lab.get("found"):
        return []
    src = {"db": "openFDA", "set_id": lab.get("set_id"), "retrieved": lab.get("retrieved")}
    classes = sorted(drug_classes(drug))
    ci = (lab.get("contraindications") or "").lower()
    boxed = lab.get("boxed_warning")
    rules = []

    for pat in RENAL_PATTERNS:
        m = re.search(pat, ci)
        if m:
            thr = int(m.group(1))
            rules.append({
                "id": f"FDA-RENAL-{drug.upper()}-{thr}", "category": "renal_dose",
                "drug": drug, "drug_class": classes, "egfr_lt": thr,
                "action": "contraindicated",
                "evidence": _snippet(ci, m.start()), "source": src,
            })
            break

    if "pregnan" in ci:
        m = re.search(r"pregnan\w*", ci)
        rules.append({
            "id": f"FDA-CONTRA-{drug.upper()}-PREGNANCY", "category": "contraindication",
            "drug": drug, "drug_class": classes, "condition": "pregnancy",
            "action": "contraindicated",
            "evidence": _snippet(ci, m.start()), "source": src,
        })

    if boxed:
        rules.append({
            "id": f"FDA-BOXED-{drug.upper()}", "category": "boxed_warning",
            "drug": drug, "drug_class": classes, "action": "caution",
            "evidence": re.sub(r"\s+", " ", boxed[:200]).strip(), "source": src,
        })
    return rules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="provena/rules/drug_safety_fda.json")
    ap.add_argument("--drugs", nargs="*", default=SEED_DRUGS)
    args = ap.parse_args()

    all_rules, missing = [], []
    for d in args.drugs:
        rs = extract_rules(d)
        if not rs and not fda_label(d).get("found"):
            missing.append(d)
        all_rules.extend(rs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": "0.1-fda",
        "description": "Drug-safety rules extracted from official FDA labels via openFDA. "
                       "Each rule cites its FDA set_id and retrieval date.",
        "rules": all_rules,
    }, indent=2))

    by_cat: dict[str, int] = {}
    for r in all_rules:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"extracted {len(all_rules)} FDA-sourced rules from {len(args.drugs)} drugs "
          f"-> {out}")
    print(f"by category: {by_cat}")
    if missing:
        print(f"no FDA label found: {missing}")
    print("\nexamples:")
    for r in all_rules:
        if r["category"] in {"renal_dose", "contraindication"}:
            print(f"  [{r['id']}] set_id={str(r['source']['set_id'])[:8]}.. :: {r['evidence'][:90]}")


if __name__ == "__main__":
    main()
