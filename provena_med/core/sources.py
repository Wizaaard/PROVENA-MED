"""Authoritative source clients for PROVENA-MED drug-safety rules.

RxNorm/RxClass (NLM) for drug normalization + ATC class membership, and openFDA for
official FDA label fields (boxed warnings, contraindications, drug interactions).
Every retrieval is cached to disk with its source identifier and retrieval date so
each derived rule is independently verifiable. Nothing here is authored from memory.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "sources_cache"
RXNAV = "https://rxnav.nlm.nih.gov/REST"
OPENFDA = "https://api.fda.gov/drug/label.json"


def _get_json(url: str, timeout: int = 15) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}


def _cache_path(kind: str, key: str) -> Path:
    safe = urllib.parse.quote(key.lower(), safe="")
    d = CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.json"


def rxnorm_normalize(name: str) -> dict:
    """Drug name -> RxCUI + ATC classes (NLM RxNorm / RxClass). Cached."""
    cp = _cache_path("rxnorm", name)
    if cp.exists():
        return json.loads(cp.read_text())
    rec: dict = {"query": name, "retrieved": dt.date.today().isoformat(),
                 "source": "NLM RxNorm/RxClass (rxnav.nlm.nih.gov)"}
    cui_js = _get_json(f"{RXNAV}/rxcui.json?name={urllib.parse.quote(name)}&search=2")
    cuis = (cui_js or {}).get("idGroup", {}).get("rxnormId", []) if cui_js else []
    rec["rxcui"] = cuis[0] if cuis else None
    rec["atc"] = []
    if rec["rxcui"]:
        cls = _get_json(
            f"{RXNAV}/rxclass/class/byRxcui.json?rxcui={rec['rxcui']}&relaSource=ATC"
        )
        for item in (cls or {}).get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
            mc = item.get("rxclassMinConceptItem", {})
            entry = {"classId": mc.get("classId"), "className": mc.get("className")}
            if entry not in rec["atc"]:
                rec["atc"].append(entry)
    cp.write_text(json.dumps(rec, indent=2))
    return rec


def fda_label(name: str) -> dict:
    """Official FDA label fields for a generic drug name (openFDA). Cached."""
    cp = _cache_path("fda", name)
    if cp.exists():
        return json.loads(cp.read_text())
    rec: dict = {"query": name, "retrieved": dt.date.today().isoformat(),
                 "source": "openFDA drug/label (api.fda.gov)"}
    q = urllib.parse.quote(f'openfda.generic_name:"{name}"')
    js = _get_json(f"{OPENFDA}?search={q}&limit=1")
    results = (js or {}).get("results", [])
    if results:
        r0 = results[0]
        rec["set_id"] = r0.get("set_id") or r0.get("id")
        rec["effective_time"] = r0.get("effective_time")
        for field in ["boxed_warning", "contraindications", "drug_interactions"]:
            val = r0.get(field)
            rec[field] = (val[0] if isinstance(val, list) and val else None)
        rec["found"] = True
    else:
        rec["found"] = False
        rec["_raw_error"] = js.get("error") if isinstance(js, dict) else None
    cp.write_text(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    drugs = ["metformin", "ibuprofen", "lisinopril", "oxycodone", "lorazepam", "warfarin"]
    print("=== RxNorm normalization (NLM) ===")
    for d in drugs:
        r = rxnorm_normalize(d)
        atc = "; ".join(f"{a['className']}" for a in r["atc"][:3]) or "(none)"
        print(f"  {d:12s} rxcui={r['rxcui']}  ATC: {atc}")
    print("\n=== openFDA label fields (FDA) ===")
    for d in drugs:
        r = fda_label(d)
        if not r.get("found"):
            print(f"  {d:12s} (no label found)")
            continue
        bw = "YES" if r.get("boxed_warning") else "no"
        ci = (r.get("contraindications") or "")[:140].replace("\n", " ")
        print(f"  {d:12s} set_id={r.get('set_id','?')[:8]}.. boxed_warning={bw}")
        print(f"               contraindications: {ci}")
