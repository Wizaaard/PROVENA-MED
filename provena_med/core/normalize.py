"""ATC-based drug normalization for PROVENA-MED (credible, source-traceable).

Drug -> internal safety class via WHO ATC class membership, fetched from NLM RxClass.
The only authored content is the internal-class -> ATC-code correspondence below, which
is a transparent mapping to the WHO ATC classification (verifiable, not a clinical
assertion). Class rosters are pulled from RxClass and cached.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "sources_cache" / "atc_members"
RXCLASS = "https://rxnav.nlm.nih.gov/REST/rxclass/classMembers.json"

# internal class -> WHO ATC code(s). Comments give the official ATC class name.
INTERNAL_CLASS_ATC: dict[str, list[str]] = {
    "nsaid": ["M01A"],                       # Antiinflammatory/antirheumatic, non-steroids
    "acei": ["C09A", "C09B"],                # ACE inhibitors (plain / combinations)
    "arb": ["C09C", "C09D"],                 # Angiotensin II receptor blockers
    "k_sparing_diuretic": ["C03D"],          # Potassium-sparing agents
    "potassium_supplement": ["A12BA"],       # Potassium
    "vka": ["B01AA"],                        # Vitamin K antagonists (warfarin)
    "heparin": ["B01AB"],                    # Heparin group
    "doac": ["B01AE", "B01AF"],              # Direct thrombin / factor Xa inhibitors
    "antiplatelet": ["B01AC"],               # Platelet aggregation inhibitors excl. heparin
    "ssri": ["N06AB"],                       # Selective serotonin reuptake inhibitors
    "maoi": ["N06AF", "N06AG"],              # Monoamine oxidase inhibitors
    "triptan": ["N02CC"],                    # Selective serotonin (5HT1) agonists
    "statin": ["C10AA"],                     # HMG CoA reductase inhibitors
    "macrolide": ["J01FA"],                  # Macrolides
    "azole_antifungal": ["J02AC"],           # Triazole/tetrazole derivatives
    "metformin": ["A10BA"],                  # Biguanides
    "nitrofurantoin": ["J01XE"],             # Nitrofuran derivatives
    "beta_blocker_nonselective": ["C07AA"],  # Beta blocking agents, non-selective
    "opioid": ["N02A"],                      # Opioids
    "benzodiazepine": ["N05BA", "N05CD", "N03AE"],  # anxiolytic/hypnotic benzos; clonazepam
    "metoclopramide": ["A03FA"],             # Propulsives (metoclopramide)
    "methotrexate": ["L01BA", "L04AX"],      # Folic acid analogues / other immunosuppressants
    "gabapentinoid": ["N02BF", "N03AX"],     # Gabapentinoids
    "loop_diuretic": ["C03C"],               # High-ceiling diuretics
    "thiazolidinedione": ["A10BG"],          # Thiazolidinediones
    "nondihydropyridine_ccb": ["C08D"],      # Selective CCBs with direct cardiac effects
    "z_drug": ["N05CF"],                     # Benzodiazepine related drugs (Z-drugs)
    "h2_blocker": ["A02BA"],                 # H2-receptor antagonists
    "tca": ["N06AA"],                        # Non-selective monoamine reuptake inhibitors (TCAs)
    "antipsychotic": ["N05A"],               # Antipsychotics (note: N05A includes lithium N05AN)
    "penicillin": ["J01C"],                  # Beta-lactam antibacterials, penicillins (allergy)
    "cephalosporin": ["J01D"],               # Other beta-lactams, cephalosporins (allergy)
    "sulfonamide": ["J01E"],                 # Sulfonamides and trimethoprim (allergy)
}


def _get_json(url: str, timeout: int = 20) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def _members_for_atc(atc: str) -> list[dict]:
    js = _get_json(f"{RXCLASS}?classId={atc}&relaSource=ATC")
    members = js.get("drugMemberGroup", {}).get("drugMember", [])
    out = []
    for m in members:
        c = m.get("minConcept", {})
        if c.get("name"):
            out.append({"rxcui": c.get("rxcui"), "name": c["name"].lower()})
    return out


def build_class_members(refresh: bool = False) -> dict[str, dict]:
    """For each internal class, fetch + cache its ATC member roster from RxClass."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rosters: dict[str, dict] = {}
    for cls, atcs in INTERNAL_CLASS_ATC.items():
        cp = CACHE / f"{cls}.json"
        if cp.exists() and not refresh:
            rosters[cls] = json.loads(cp.read_text())
            continue
        names, rxcuis = set(), set()
        for atc in atcs:
            for m in _members_for_atc(atc):
                names.add(m["name"])
                if m["rxcui"]:
                    rxcuis.add(m["rxcui"])
        rec = {"class": cls, "atc": atcs, "retrieved": dt.date.today().isoformat(),
               "source": "NLM RxClass (WHO ATC membership)",
               "names": sorted(names), "rxcuis": sorted(rxcuis)}
        cp.write_text(json.dumps(rec, indent=2))
        rosters[cls] = rec
    return rosters


_ROSTERS: dict[str, dict] | None = None
_NAME_INDEX: list[tuple[str, str]] | None = None  # (member_name, class)


def _ensure_index() -> None:
    global _ROSTERS, _NAME_INDEX
    if _ROSTERS is None:
        _ROSTERS = build_class_members()
        _NAME_INDEX = []
        for cls, rec in _ROSTERS.items():
            for nm in rec["names"]:
                if len(nm) >= 4:  # avoid spurious short-token matches
                    _NAME_INDEX.append((nm, cls))


@lru_cache(maxsize=20000)
def drug_classes(med_text: str) -> set[str]:
    """Map a free-text medication string to internal safety classes via ATC rosters.
    Cached: pure + meds repeat heavily across cases (callers must not mutate the result)."""
    _ensure_index()
    s = med_text.lower()
    out: set[str] = set()
    for member_name, cls in _NAME_INDEX:  # type: ignore[union-attr]
        if re.search(rf"\b{re.escape(member_name)}\b", s):
            out.add(cls)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    rosters = build_class_members()
    print("=== ATC class rosters (from NLM RxClass) ===")
    for cls in ["nsaid", "opioid", "benzodiazepine", "acei", "statin", "vka"]:
        r = rosters[cls]
        print(f"  {cls:14s} ATC={r['atc']}  members={len(r['names'])}  e.g. {r['names'][:5]}")

    # coverage comparison on the proposed meds from the safety experiment
    from score_safety import parse_plan
    from provena_med.core import safety as hand
    out_file = Path(__file__).resolve().parent.parent / "outputs" / "exp0safety_llama31_8b.jsonl"
    if out_file.exists():
        meds = []
        for line in open(out_file):
            _, m = parse_plan(json.loads(line)["output"])
            meds += m
        uniq = sorted(set(m.lower() for m in meds))
        hand_cov = sum(1 for m in meds if hand.classes_of(m))
        atc_cov = sum(1 for m in meds if drug_classes(m))
        print(f"\n=== coverage on {len(meds)} proposed meds ({len(uniq)} unique) ===")
        print(f"  hand lexicon : {hand_cov}/{len(meds)} = {hand_cov/len(meds):.1%}")
        print(f"  ATC/RxClass  : {atc_cov}/{len(meds)} = {atc_cov/len(meds):.1%}")
