"""PROVENA-MED staged, provenance-linked generation (phase 1).

The model sees the patient context AND a bundle of typed evidence units with stable IDs,
and must produce the 5-stage artifact where every claim cites evidence IDs. Saves the
bundle alongside the output so MM-AIS can be scored later (phase 2).

  --cohort ed          text cohort (ED): NOTE_SPAN + TABLE_ROW
  --cohort cardiac_mm  multimodal (cardiac): NOTE_SPAN + IMAGE_FINDING (report text)
  --cohort cxr_pixel   true-pixel (cardiac x CXR): NOTE_SPAN + IMAGE_FINDING read off
                       real radiograph pixels by a vision model (--pixel-findings cache)
  --cohort icu_mm      multimodal ICU (MIMIC-IV icu): NOTE_SPAN (HPI) + TABLE_ROW
                       (first-24h labs+vitals) + IMAGE_FINDING (CXR pixels; --pixel-findings)

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_generate_staged.py \
      --cohort cardiac_mm --n 50 --out outputs/staged_cardiac_mm_llama31_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import (  # noqa: E402
    build_bundle, build_bundle_cxr_pixel, build_bundle_eicu, build_bundle_icu,
    build_bundle_mimic3, build_bundle_mm, bundle_to_prompt)
from provena_med.core.edcds import _parse_dx_field  # noqa: E402
from provena_med.core.guidelines import load_catalog  # noqa: E402

PROVENA_DIR = Path("<DATA_ROOT>/PROVENA-MED/v0.2/cohorts")
_CATALOG: dict | None = None


def _guideline_units(ids) -> list[dict]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = {u["id"]: u for u in load_catalog()}
    return [_CATALOG[i] for i in (ids or []) if i in _CATALOG]

DEFAULT_MODEL = "~/models/Llama-3.1-8B-Instruct"
PERSONA = "You are an experienced emergency medicine physician."
SCHEMA = (
    '{\n'
    '  "problem_list": [{"claim": "...", "evidence": ["ID", ...]}],\n'
    '  "differential": [{"diagnosis": "...", "rank": 1, "evidence": ["ID", ...]}],\n'
    '  "workup": [{"action": "...", "evidence": ["ID", ...]}],\n'
    '  "therapy": [{"action": "...", "evidence": ["ID", ...], "safety_status": "allowed|caution|contraindicated|insufficient"}],\n'
    '  "monitoring": [{"action": "...", "evidence": ["ID", ...]}],\n'
    '  "missing_information": ["..."]\n'
    '}'
)


def _load_pixel_findings(path: str) -> dict:
    """hadm_id -> list[str] of pixel-derived IMAGE_FINDING texts (from run_cxr_findings.py)."""
    findings = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            findings[int(r["id"])] = r.get("image_findings", [])
    return findings


def load_cases(cohort: str, n: int, seed: int, pixel_findings: str | None = None) -> list[dict]:
    if cohort == "ed":
        from run_generate_safety import load_safety_cases
        df = load_safety_cases(n, seed)
        out = []
        for _, row in df.iterrows():
            c = row.to_dict()
            out.append({
                "id": int(c["stay_id"]),
                "gold": {"primary": _parse_dx_field(c["primary_diagnosis"]),
                         "secondary": _parse_dx_field(c["secondary_diagnosis"])},
                "header": (f"Demographics: {str(c.get('patient_info','')).strip()}\n"
                           f"Initial vitals: {str(c.get('initial_vitals','')).strip()}"),
                "bundle": build_bundle(c),
            })
        return out
    elif cohort == "cardiac_mm":
        from provena_med.core.cohort_mm import load_mm_cases
        out = []
        for c in load_mm_cases(n, seed):
            out.append({
                "id": int(c["id"]),
                "gold": {"primary": c["gold"], "secondary": []},
                "header": f"Chief complaint: {c.get('chief_complaint','').strip()}",
                "bundle": build_bundle_mm(c),
            })
        return out
    elif cohort == "cxr_pixel":
        if not pixel_findings:
            raise ValueError("--cohort cxr_pixel requires --pixel-findings <cache.jsonl> "
                             "(produce it with run_cxr_findings.py)")
        from provena_med.core.cohort_cxr import load_cxr_cases
        pf = _load_pixel_findings(pixel_findings)
        out = []
        for c in load_cxr_cases(n, seed):
            out.append({
                "id": int(c["id"]),
                "gold": {"primary": c["gold"], "secondary": []},
                "header": f"Chief complaint: {str(c.get('chief_complaint','')).strip()}",
                "bundle": build_bundle_cxr_pixel(c, pf.get(int(c["id"]), [])),
            })
        return out
    elif cohort == "icu_mm":
        from provena_med.core.cohort_icu import load_icu_cases
        pf = _load_pixel_findings(pixel_findings) if pixel_findings else {}
        out = []
        for c in load_icu_cases(n, seed):
            out.append({
                "id": int(c["id"]),
                "gold": {"primary": c["gold"], "secondary": []},
                "header": (f"Chief complaint: {str(c.get('chief_complaint','')).strip()}\n"
                           f"ICU admission; evidence is from the first 24h of the stay."),
                "bundle": build_bundle_icu(c, pf.get(int(c["id"]), [])),
            })
        return out
    elif cohort == "eicu":
        from provena_med.core.cohort_eicu import load_eicu_cases
        out = []
        for c in load_eicu_cases(n, seed):
            out.append({
                "id": int(c["id"]),
                "gold": {"primary": c["gold"], "secondary": []},
                "header": (f"External ICU stay (eICU-CRD). Patient: {c.get('demographics','')}."),
                "bundle": build_bundle_eicu(c),
            })
        return out
    elif cohort == "mimic3":
        from provena_med.core.cohort_mimic3 import load_mimic3_cases
        out = []
        for c in load_mimic3_cases(n, seed):
            out.append({
                "id": int(c["id"]),
                "gold": {"primary": c["gold"], "secondary": []},
                "header": (f"External admission (MIMIC-III). "
                           f"Chief complaint: {str(c.get('chief_complaint','')).strip()}"),
                "bundle": build_bundle_mimic3(c),
            })
        return out
    raise ValueError(cohort)


_BUILDERS = {"ed": lambda c, pf: build_bundle(c),
             "cardiac_mm": lambda c, pf: build_bundle_mm(c),
             # default IMAGE_FINDING = radiologist report (on the case); --pixel-findings
             # overrides with a VLM's reading for a vision baseline.
             "icu_mm": lambda c, pf: build_bundle_icu(c, pf.get(int(c["id"])) if pf else None),
             "eicu": lambda c, pf: build_bundle_eicu(c),
             "mimic3": lambda c, pf: build_bundle_mimic3(c)}


def load_provena(cohort: str, n: int, seed: int, pixel_findings: str | None = None) -> list[dict]:
    """Load from the canonical PROVENA-MED dataset (v0.3): full bundle incl. the case's
    applicable GUIDELINE_RULE units, so therapy claims can cite guidelines."""
    import random
    recs = [json.loads(line) for line in open(PROVENA_DIR / f"{cohort}.jsonl")]
    if n and n < len(recs):
        recs = [recs[i] for i in sorted(random.Random(seed).sample(range(len(recs)), n))]
    pf = _load_pixel_findings(pixel_findings) if pixel_findings else {}
    build = _BUILDERS[cohort]
    out = []
    for c in recs:
        bundle = build(c, pf) + _guideline_units(c.get("guideline_rules"))
        cc = str(c.get("chief_complaint", "")).strip()
        demo = str(c.get("demographics", "")).strip()
        header = (f"Cohort: {cohort}. " + (f"Chief complaint: {cc}. " if cc else "")
                  + (f"Patient: {demo}." if demo else "")).strip()
        out.append({"id": int(c["id"]), "gold": {"primary": c["gold"], "secondary": []},
                    "header": header, "bundle": bundle})
    return out


def build_messages(header: str, units: list[dict]) -> list[dict]:
    user = (
        f"{header}\n\n"
        "EVIDENCE UNITS (cite these IDs):\n"
        f"{bundle_to_prompt(units)}\n\n"
        "Produce a staged clinical reasoning artifact. For EVERY claim, cite one or more "
        "evidence IDs from the list above (use the exact IDs; cite only listed IDs). For "
        "therapy/safety claims, also cite any applicable GUIDELINE_RULE IDs that justify the "
        "choice or its safety_status. If a stage cannot be supported by the available "
        "evidence, return an empty list for it and add a note to missing_information. "
        "Return ONLY this JSON object:\n"
        f"{SCHEMA}"
    )
    return [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cohort", default="ed",
                    choices=["ed", "cardiac_mm", "cxr_pixel", "icu_mm", "eicu", "mimic3"])
    ap.add_argument("--provena", action="store_true",
                    help="load from the canonical PROVENA-MED v0.3 dataset (incl. GUIDELINE_RULE units)")
    ap.add_argument("--pixel-findings", default=None,
                    help="JSONL of pixel-derived IMAGE_FINDING units (for --cohort cxr_pixel)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.provena:
        cases = load_provena(args.cohort, args.n, args.seed, args.pixel_findings)
    else:
        cases = load_cases(args.cohort, args.n, args.seed, args.pixel_findings)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def render(messages):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates that forbid a separate system role (e.g. BioMistral)
            merged, sys_txt = [], ""
            for m in messages:
                if m["role"] == "system":
                    sys_txt = m["content"]
                elif m["role"] == "user":
                    merged.append({"role": "user", "content": (sys_txt + "\n\n" + m["content"]).strip()})
                    sys_txt = ""
                else:
                    merged.append(m)
            return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    prompts = [render(build_messages(c["header"], c["bundle"])) for c in cases]
    outputs: list[str] = []
    for s in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[s:s + args.batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"[staged:{args.cohort}] {min(s + args.batch_size, len(prompts))}/{len(prompts)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c, text in zip(cases, outputs):
            f.write(json.dumps({"stay_id": c["id"], "gold": c["gold"],
                                "bundle": c["bundle"], "output": text}) + "\n")
    print(f"[staged:{args.cohort}] wrote {len(cases)} -> {out}")


if __name__ == "__main__":
    main()
