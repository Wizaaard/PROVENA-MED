"""PROVENA-MED drug-safety checker (sourced rules only).

Loads rules transcribed from Phansalkar 2012 + Beers 2023 (sourced_rules.json) and
derived from FDA labels (drug_safety_fda.json). Drug->class normalization uses WHO ATC
via NLM RxClass (provena/normalize.py). No rule logic is authored from memory.

Public API (used by score_safety.py): load_rules, extract_context, check, classes_of.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from provena_med.core import normalize

RULES_DIR = Path(__file__).resolve().parent / "rules"

CONDITION_PATTERNS = {
    "ckd": ["chronic kidney", "ckd", "esrd", "end stage renal", "renal failure",
            "renal insufficiency", "dialysis", "hemodialysis", "ckd stage", "egfr", "gfr <"],
    "chf": ["heart failure", "chf", "congestive heart", "cardiomyopathy", "reduced ejection", "hfref", "hfpef"],
    "asthma": ["asthma"],
    "copd": ["copd", "emphysema"],
    "pregnancy": ["pregnan", "gravid", "gestation"],
    "hyperkalemia": ["hyperkalemia", "hyperkalemic"],
    "cad": ["coronary artery", "myocardial infarction", "angina", "stemi", "nstemi", " cad "],
    "parkinson": ["parkinson"],
    "delirium": ["delirium", "altered mental status", "acute confusion", "encephalopath"],
    "dementia": ["dementia", "alzheimer", "cognitive impairment", "cognitive decline"],
    "falls": ["fell", "mechanical fall", "history of fall", "recurrent fall", "fall from",
              "s/p fall", "status post fall", "found down", "fracture"],
    "peptic_ulcer": ["peptic ulcer", "gastric ulcer", "duodenal ulcer", "pud", "gi bleed",
                     "gastrointestinal bleed", "upper gi bleed"],
}


def classes_of(med_name: str) -> set[str]:
    return normalize.drug_classes(med_name)


@lru_cache(maxsize=1)
def load_rules() -> list[dict]:
    rules = json.loads((RULES_DIR / "sourced_rules.json").read_text())["rules"]
    # merge FDA-derived rules (normalize their schema; skip informational boxed warnings)
    fda_path = RULES_DIR / "drug_safety_fda.json"
    if fda_path.exists():
        for r in json.loads(fda_path.read_text())["rules"]:
            sid = (r.get("source") or {}).get("set_id")
            src = f"openFDA label (set_id {sid})"
            if r["category"] == "renal_dose":
                rules.append({"id": r["id"], "category": "renal_dose", "members": [r["drug"]],
                              "egfr_lt": r["egfr_lt"], "action": r["action"],
                              "message": r.get("evidence", ""), "source": src})
            elif r["category"] == "contraindication":
                rules.append({"id": r["id"], "category": "contraindication", "condition": r["condition"],
                              "members": [r["drug"]], "action": r["action"],
                              "message": r.get("evidence", ""), "source": src})
    return rules


# ---------------- context extraction ----------------
def ckd_epi_egfr(scr: float, age: int, female: bool) -> float:
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    sex_f = 1.012 if female else 1.0
    return 142 * (min(scr / kappa, 1) ** alpha) * (max(scr / kappa, 1) ** -1.200) * (0.9938 ** age) * sex_f


def _creatinine(tests: str) -> float | None:
    if not tests:
        return None
    for pat in [r"creat[a-z]*[\s:\-]+(\d\.?\d*)", r"\bscr\b[\s:\-]+(\d\.?\d*)", r"\bcr[\s:\-]+(\d\.?\d*)"]:
        m = re.search(pat, tests.lower())
        if m:
            v = float(m.group(1))
            if 0.1 < v < 25:
                return v
    return None


def _allergy(full_text: str) -> tuple[str, set]:
    """Extract the Allergies block from a discharge note: raw text + mapped classes."""
    if not full_text:
        return "", set()
    low = full_text.lower()
    m = re.search(
        r"allerg(?:y|ies)[^:\n]*:\s*(.*?)(?:\n\s*\n|\nattending|\nchief complaint|"
        r"\nmajor surg|\nhistory of present|\npast medical|\nmedications on admission|\nphysical exam)",
        low, flags=re.DOTALL)
    if not m:
        m = re.search(r"allerg(?:y|ies)[^:\n]*:\s*(.{0,160})", low, flags=re.DOTALL)
    if not m:
        return "", set()
    seg = m.group(1).strip()
    if not seg or "no known" in seg or "nka" in seg.replace(".", "") or seg.startswith("none"):
        return "", set()
    return seg, normalize.drug_classes(seg)


def _demo(patient_info: str) -> tuple[int | None, bool | None]:
    s = str(patient_info).lower()
    age = int(re.search(r"age[:\s]+(\d{1,3})", s).group(1)) if re.search(r"age[:\s]+(\d{1,3})", s) else None
    female = True if "female" in s else (False if "male" in s else None)
    return age, female


def _conditions(*texts: str) -> set[str]:
    blob = " ".join(str(t).lower() for t in texts if t)
    return {tag for tag, pats in CONDITION_PATTERNS.items() if any(p in blob for p in pats)}


def _med_tokens(text: str) -> list[str]:
    out = []
    for line in re.split(r"[\n;]+", str(text)):
        line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if line and line.lower() != "nan":
            out.append(line)
    return out


def extract_context(case: dict) -> dict:
    age, female = _demo(case.get("patient_info", ""))
    scr = _creatinine(case.get("tests", ""))
    egfr = ckd_epi_egfr(scr, age, female) if (scr and age is not None and female is not None) else None
    current = _med_tokens(case.get("past_medication", ""))
    allergy_text, allergy_classes = _allergy(case.get("text", "") or "")
    return {
        "age": age, "egfr": egfr, "creatinine": scr,
        "conditions": _conditions(case.get("HPI", ""), case.get("text", "")),
        "allergy_text": allergy_text,
        "allergy_classes": allergy_classes,
        "current_tokens": current,
        "current_classes": {c for t in current for c in normalize.drug_classes(t)},
    }


# ---------------- rule evaluation ----------------
def _present(members: list, tokens: list[str], classes: set[str]) -> str | None:
    low = [t.lower() for t in tokens]
    for m in members:
        if isinstance(m, dict):
            if m.get("atc") in classes:
                return m["atc"]
        else:
            ml = m.lower()
            for t in low:
                if re.search(rf"\b{re.escape(ml)}\b", t):
                    return m
    return None


def check(proposed_meds: list[str], ctx: dict, rules: list[dict]) -> list[dict]:
    prop_tokens = list(proposed_meds)
    prop_classes = {c for m in proposed_meds for c in normalize.drug_classes(m)}
    pool_tokens = prop_tokens + ctx["current_tokens"]
    pool_classes = prop_classes | ctx["current_classes"]
    age = ctx.get("age")
    violations = []

    for r in rules:
        if r.get("age_min") is not None and (age is None or age < r["age_min"]):
            continue
        cond = r.get("condition")
        if cond and cond not in ctx["conditions"]:
            continue
        cat = r["category"]
        if cat == "renal_dose":
            if ctx["egfr"] is None:
                continue
            hit = _present(r["members"], prop_tokens, prop_classes)
            if hit and ctx["egfr"] < r["egfr_lt"]:
                violations.append({"rule": r["id"], "drug": hit, "action": r["action"],
                                   "message": r.get("message", ""), "source": r.get("source", "")})
        elif cat == "contraindication":
            hit = _present(r["members"], prop_tokens, prop_classes)
            if hit:
                violations.append({"rule": r["id"], "drug": hit, "action": r["action"],
                                   "message": r.get("message", ""), "source": r.get("source", "")})
        elif cat == "ddi":
            a_prop = _present(r["members_a"], prop_tokens, prop_classes)
            b_prop = _present(r["members_b"], prop_tokens, prop_classes)
            a_pool = a_prop or _present(r["members_a"], ctx["current_tokens"], ctx["current_classes"])
            b_pool = b_prop or _present(r["members_b"], ctx["current_tokens"], ctx["current_classes"])
            if a_pool and b_pool and (a_prop or b_prop):
                violations.append({"rule": r["id"], "drug": f"{a_pool}+{b_pool}", "action": r["action"],
                                   "message": r.get("message", ""), "source": r.get("source", "")})

    # allergy: class cross-reactivity OR direct drug-name match against the allergy list
    atext = ctx.get("allergy_text", "")
    seen: set[str] = set()
    for med in proposed_meds:
        hit = None
        inter = normalize.drug_classes(med) & ctx["allergy_classes"]
        if inter:
            hit = next(iter(inter))
        elif atext:
            for tok in re.findall(r"[a-z]{5,}", med.lower()):
                if re.search(rf"\b{re.escape(tok)}\b", atext):
                    hit = tok
                    break
        if hit and hit not in seen:
            seen.add(hit)
            violations.append({"rule": "ALLERGY-MATCH", "drug": med, "action": "contraindicated",
                               "message": f"Proposed drug matches a documented patient allergy ({hit}).",
                               "source": "patient allergy record"})
    return violations
