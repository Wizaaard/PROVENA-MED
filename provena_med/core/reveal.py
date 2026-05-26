"""Interactive reveal schema for PROVENA-MED (evidence-seeking track, all cohorts).

The benchmark's interactive track presents an under-specified picture and lets the agent
*request* more evidence. Because every evidence unit already has a typed, stable ID, the
reveal schema is a pure function of the bundle: units are bucketed by ID prefix into what
is visible at start (s0) vs. what each clinical ACTION reveals. This generalizes the old
ED-only harness to ICU / cardiac / eICU / MIMIC-III with no per-cohort code.

  s0 (free)            : initial vitals (chief complaint + demographics live in the header)
  take_history         : HPI + past-history note spans
  physical_exam        : exam note spans
  order_labs           : laboratory table rows
  order_imaging        : image findings
  consult_guidelines   : applicable GUIDELINE_RULE units
"""
from __future__ import annotations

# (id-prefix, action). "_s0" = revealed for free at the start.
ACTION_OF = [
    ("NOTE_SPAN:hpi", "take_history"),
    ("NOTE_SPAN:hx", "take_history"),
    ("NOTE_SPAN:med", "med_reconciliation"),
    ("NOTE_SPAN:exam", "physical_exam"),
    ("TABLE_ROW:vital", "_s0"),
    ("TABLE_ROW:lab", "order_labs"),
    ("IMAGE_FINDING", "order_imaging"),
    ("GUIDELINE_RULE", "consult_guidelines"),
]
ACTION_COST = {"take_history": 1, "med_reconciliation": 1, "physical_exam": 1,
               "order_labs": 2, "order_imaging": 3, "consult_guidelines": 0}


def _bucket(uid: str) -> str:
    for prefix, act in ACTION_OF:
        if uid.startswith(prefix):
            return act
    return "_s0"


def build_reveal(bundle: list[dict]) -> dict:
    """{s0_visible:[ids], actions:{action:[ids]}, action_cost:{action:int}}."""
    s0, actions = [], {}
    for u in bundle:
        act = _bucket(u["id"])
        if act == "_s0":
            s0.append(u["id"])
        else:
            actions.setdefault(act, []).append(u["id"])
    return {"s0_visible": s0, "actions": actions,
            "action_cost": {a: ACTION_COST.get(a, 1) for a in actions}}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from provena_med.core.cohort_icu import load_icu_cases
    from provena_med.core.bundle import build_bundle_icu
    from provena_med.core.guidelines import relevant_units
    c = load_icu_cases(1, 0)[0]
    b = build_bundle_icu(c, ["Cardiomegaly with pulmonary edema.", "Small pleural effusion."])
    b += relevant_units({"chf"}, egfr=None, age=80)  # demo: add applicable guideline units
    r = build_reveal(b)
    print("s0_visible:", len(r["s0_visible"]), "units")
    for a, ids in r["actions"].items():
        print(f"  action {a:18} cost={r['action_cost'][a]}  ->  {len(ids)} units  e.g. {ids[:2]}")
