"""MM-AIS: modality-stratified attribution scoring for PROVENA-MED.

For each generated claim with cited evidence IDs, an independent clinical NLI verifier
judges whether each cited unit entails/supports the claim. We report citation precision
(fraction of citations that are supportive), stratified by modality, plus citation
validity and coverage. Recall (gold-required evidence) and the counterfactual
necessity/sufficiency tests are deferred.
"""
from __future__ import annotations

import json
import re

import numpy as np

NLI_MODEL = "pritamdeka/PubMedBERT-MNLI-MedNLI"
STAGES = ["problem_list", "differential", "workup", "therapy", "monitoring"]
_CLAIM_KEYS = ["claim", "diagnosis", "action"]


def _try_load(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", s)  # trailing commas
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def parse_staged(text: str) -> dict | None:
    """Extract the staged JSON artifact from a model response (lenient)."""
    text = re.sub(r"```(?:json)?", " ", text)  # strip markdown fences
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                obj = _try_load(text[start:i + 1])
                if isinstance(obj, dict) and any(s in obj for s in STAGES):
                    return obj
                start = None
    return None


def extract_claims(artifact: dict) -> list[dict]:
    """Flatten the artifact into {stage, claim_text, evidence_ids}."""
    claims = []
    for stage in STAGES:
        for item in artifact.get(stage, []) or []:
            if not isinstance(item, dict):
                continue
            text = next((str(item[k]) for k in _CLAIM_KEYS if item.get(k)), None)
            if not text:
                continue
            ev = item.get("evidence", []) or []
            ev = [str(e).strip() for e in ev if str(e).strip()]
            claims.append({"stage": stage, "claim": text, "evidence": ev})
    return claims


def _grab_object(s: str, i: int) -> tuple[str | None, int]:
    """If s[i]=='{', return (balanced_object_str, end_idx); (None, len) if truncated."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i:j + 1], j + 1
    return None, len(s)


def extract_claims_lenient(text: str) -> list[dict]:
    """Recover {stage, claim, evidence} from possibly-truncated staged JSON.

    Scans each stage's array and collects every *complete* claim object, so truncated
    outputs (verbose models hitting the token limit) still yield their finished claims.
    """
    text = re.sub(r"```(?:json)?", " ", text)
    claims: list[dict] = []
    for stage in STAGES:
        m = re.search(rf'"{re.escape(stage)}"\s*:\s*\[', text)
        if not m:
            continue
        i, depth_arr = m.end(), 1
        while i < len(text) and depth_arr > 0:
            ch = text[i]
            if ch == "{":
                obj_str, nxt = _grab_object(text, i)
                if obj_str is None:
                    break
                obj = _try_load(obj_str)
                if isinstance(obj, dict):
                    t = next((str(obj[k]) for k in _CLAIM_KEYS if obj.get(k)), None)
                    if t:
                        ev = [str(e).strip() for e in (obj.get("evidence") or []) if str(e).strip()]
                        claims.append({"stage": stage, "claim": t, "evidence": ev})
                i = nxt
            elif ch == "[":
                depth_arr += 1
                i += 1
            elif ch == "]":
                depth_arr -= 1
                i += 1
            else:
                i += 1
    return claims


class NLIVerifier:
    def __init__(self, model_name: str = NLI_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent_idx = next((i for i, l in id2label.items() if "entail" in l), 0)

    def entail(self, premises: list[str], hypotheses: list[str], bs: int = 64) -> list[float]:
        probs = []
        for s in range(0, len(premises), bs):
            enc = self.tok(premises[s:s + bs], hypotheses[s:s + bs], truncation=True,
                           padding=True, max_length=256, return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                logit = self.model(**enc).logits
            p = logit.softmax(-1)[:, self.ent_idx].tolist()
            probs.extend(p)
        return probs


# --- provenance RECALL: did the model cite the salient evidence it was given? ---
# "Salient" is defined objectively (no annotation): abnormal labs, fired guideline rules,
# and positive (non-negative) image findings -- evidence a faithful workup must surface.
_NEG_RE = re.compile(r"\bno\b|\bnormal\b|\bclear\b|unremarkable|\bstable\b|without|"
                     r"\bnegative\b|\bnone\b|no acute", re.I)
_ABN_RE = re.compile(r"\((?:abnormal|h|l|hh|ll|high|low|critical|\*|delta)\)", re.I)


def _is_salient(u: dict) -> bool:
    t = u.get("text", "")
    if u["type"] == "GUIDELINE_RULE":
        return True  # an applicable safety rule that fired
    if u["type"] == "IMAGE_FINDING":
        return not _NEG_RE.search(t)  # a positive radiographic finding
    if u["type"] == "TABLE_ROW" and u["id"].startswith("TABLE_ROW:lab"):
        return bool(_ABN_RE.search(t))  # a flagged-abnormal lab
    return False


def score_recall(records: list[dict]) -> dict:
    """Salient-evidence recall: fraction of objectively-salient units cited anywhere in the
    artifact, overall and by modality. Reproducible (no judge)."""
    tot: dict[str, int] = {}
    hit: dict[str, int] = {}
    for rec in records:
        units = {u["id"]: u for u in rec["bundle"]}
        salient = {uid for uid, u in units.items() if _is_salient(u)}
        cited = set()
        for c in extract_claims_lenient(rec["output"]):
            cited.update(e for e in c["evidence"] if e in units)
        for uid in salient:
            mod = units[uid]["type"]
            tot[mod] = tot.get(mod, 0) + 1
            if uid in cited:
                hit[mod] = hit.get(mod, 0) + 1
    T, H = sum(tot.values()), sum(hit.values())
    return {"recall": (H / T) if T else 0.0, "n_salient": T, "n_salient_cited": H,
            "recall_by_modality": {m: hit.get(m, 0) / tot[m] for m in tot}}


def score_records(records: list[dict], tau: float = 0.5, verifier=None) -> dict:
    verifier = verifier or NLIVerifier()
    # gather all (premise=unit text, hypothesis=claim) pairs
    premises, hyps, meta = [], [], []  # meta: (rec_idx, modality)
    n_claims = n_claims_cited = n_valid = n_cited = 0
    parsed_ok = 0
    for ri, rec in enumerate(records):
        claims = extract_claims_lenient(rec["output"])
        if not claims:
            continue
        parsed_ok += 1
        units = {u["id"]: u for u in rec["bundle"]}
        for c in claims:
            n_claims += 1
            cited = c["evidence"]
            n_cited += len(cited)
            valid = [e for e in cited if e in units]
            n_valid += len(valid)
            if valid:
                n_claims_cited += 1
            for e in valid:
                premises.append(units[e]["text"])
                hyps.append(c["claim"])
                meta.append((ri, units[e]["type"]))

    if not premises:
        return {"n": len(records), "parsed": parsed_ok, "claims": n_claims,
                "claims_with_valid_citation": n_claims_cited,
                "citation_validity": (n_valid / n_cited) if n_cited else 0.0,
                "mean_citations_per_claim": (n_cited / n_claims) if n_claims else 0.0,
                "mm_ais_precision": 0.0, "mm_ais_precision_by_modality": {},
                "n_scored_citations": 0, "tau": tau}

    ent = verifier.entail(premises, hyps)
    by_mod: dict[str, list[float]] = {}
    for (_, mod), p in zip(meta, ent):
        by_mod.setdefault(mod, []).append(1.0 if p >= tau else 0.0)
    supportive = [1.0 if p >= tau else 0.0 for p in ent]

    return {
        "n": len(records), "parsed": parsed_ok, "claims": n_claims,
        "claims_with_valid_citation": n_claims_cited,
        "citation_validity": (n_valid / n_cited) if n_cited else 0.0,
        "mean_citations_per_claim": (n_cited / n_claims) if n_claims else 0.0,
        "mm_ais_precision": float(np.mean(supportive)),
        "mm_ais_precision_by_modality": {m: float(np.mean(v)) for m, v in by_mod.items()},
        "n_scored_citations": len(supportive),
        "tau": tau,
    }
