"""GUIDELINE_RULE evidence units for PROVENA-MED (curated, sourced drug-safety rules).

Turns the authoritative rules in provena/rules/ (Phansalkar 2012 DDIs, Beers 2023
drug-disease/renal, FDA labels) into citable evidence units with stable IDs, so a model's
therapy-stage claim can cite the actual guideline it relied on (GUIDELINE_RULE modality of
the MM-AIS metric). `relevant_units` selects the rules whose preconditions the patient's
state already meets (condition present, renal threshold crossed, documented allergy), so a
case's bundle carries the guidelines that actually apply -- not the whole catalog.
"""
from __future__ import annotations

from provena_med.core import safety


def _unit(r: dict) -> dict:
    act = str(r.get("action", "")).upper()
    msg = str(r.get("message", "")).strip()
    text = f"[{act}] {msg}" if msg else f"[{act}] safety rule {r['id']}"
    return {"id": f"GUIDELINE_RULE:{r['id']}", "type": "GUIDELINE_RULE", "text": text[:400],
            "source": r.get("source", ""), "category": r.get("category"),
            "action": r.get("action"), "age_min": r.get("age_min")}


def load_catalog() -> list[dict]:
    """Full citable rule library (~76 sourced GUIDELINE_RULE units)."""
    return [_unit(r) for r in safety.load_rules()]


def _members(r: dict):
    return (r.get("members") or []) + (r.get("members_a") or []) + (r.get("members_b") or [])


def relevant_units(conditions: set[str], egfr: float | None, age: int | None,
                   allergy: bool = False, current_classes: set[str] | None = None) -> list[dict]:
    """Rules whose precondition is already satisfied by the patient's state."""
    current_classes = current_classes or set()
    out = []
    for r in safety.load_rules():
        if r.get("age_min") is not None and (age is None or age < r["age_min"]):
            continue
        cat = r.get("category")
        keep = False
        if cat == "renal_dose":
            keep = egfr is not None and egfr < r.get("egfr_lt", 0)
        elif r.get("condition"):
            keep = r["condition"] in conditions
        elif cat == "ddi":
            # one side already in the patient's current medication classes
            keep = bool(current_classes) and any(
                isinstance(m, dict) and m.get("atc") in current_classes for m in _members(r))
        elif cat in ("allergy",):
            keep = allergy
        if keep:
            out.append(_unit(r))
    return out


if __name__ == "__main__":
    cat = load_catalog()
    print(f"catalog: {len(cat)} GUIDELINE_RULE units")
    for u in cat[:3]:
        print(" ", u["id"], "|", u["text"][:80], "|", u["source"][:50])
    rel = relevant_units({"chf", "ckd", "falls"}, egfr=25, age=78, allergy=False)
    print(f"\nrelevant for {{chf,ckd,falls}}, eGFR 25, age 78: {len(rel)} rules")
    for u in rel[:6]:
        print(" ", u["id"], "|", u["text"][:70])
