"""PROVENA-MED triangulation protocol: clinician-free gold corroboration.

Every gold diagnosis is cross-checked against INDEPENDENT sources, so labels used for
evaluation are corroborated without any new clinician annotation:

  CODE        the diagnosis is billed for the encounter (administrative; true by
              construction for ICD-derived gold).
  NARRATIVE   the diagnosis concept is present in the free-text notes, judged by TWO
              independent extractors -- BioLORD embedding match and an LLM-as-judge
              (Llama-3.3-70B, held-out). Their agreement (Cohen's kappa) is reported as
              extraction reliability; a label is narratively supported if either fires.
  STRUCTURED  an objective lab signature consistent with the diagnosis is met (standard
              cutoffs; applies only to diagnoses with a defined signature).

corroboration level = CODE(1) + NARRATIVE(1) + STRUCTURED(1 if applicable & met).
A label is "triangulated" (high-precision gold) if level >= 2.

  HF_HUB_OFFLINE=1 python triangulate.py --cohort icu_mm --n 200 --out outputs/triang_icu.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import split_sentences  # noqa: E402

TAU_EMB, TAU_NLI = 0.45, 0.5

# objective lab signatures (standard clinical cutoffs; each is citable, not invented dx)
DX_SIGNATURE = [
    (r"kidney|renal failure|aki|nephropathy", lambda L: L.get("creatinine", 0) > 1.5),
    (r"anemia|anaemia", lambda L: 0 < L.get("hemoglobin", 99) < 10),
    (r"hyponatremia", lambda L: 0 < L.get("sodium", 999) < 135),
    (r"hypernatremia", lambda L: L.get("sodium", 0) > 145),
    (r"hyperkalemia", lambda L: L.get("potassium", 0) > 5.0),
    (r"hypokalemia", lambda L: 0 < L.get("potassium", 9) < 3.5),
    (r"diabet|hyperglycem|ketoacidosis", lambda L: L.get("glucose", 0) > 200),
    (r"sepsis|septic|infection", lambda L: L.get("lactate", 0) > 2.0 or L.get("wbc", 0) > 12),
    (r"respiratory failure|hypox", lambda L: (0 < L.get("po2", 999) < 60) or L.get("pco2", 0) > 50),
    (r"thrombocytopenia", lambda L: 0 < L.get("platelets", 999) < 100),
    (r"leukocytosis", lambda L: L.get("wbc", 0) > 12),
    (r"acidosis", lambda L: (0 < L.get("ph", 9) < 7.35) or (0 < L.get("bicarbonate", 99) < 20)),
    (r"hepat|liver", lambda L: L.get("alt", 0) > 100 or L.get("ast", 0) > 100 or L.get("bilirubin_total", 0) > 2),
]


def cohort_records(cohort: str, n: int, seed: int) -> list[dict]:
    """Unify cohorts to {id, gold:[str], sents:[str], labs:{name:value}}."""
    def labflat(labs):  # {name: numeric value}
        out = {}
        for k, v in (labs or {}).items():
            val = v.get("value") if isinstance(v, dict) else v
            try:
                out[k] = float(val)
            except (TypeError, ValueError):
                pass
        return out

    recs = []
    if cohort == "ed":
        from provena_med.core.edcds import load_cases, gold_diagnoses, case_presentation
        df = load_cases(n, seed)
        for _, r in df.iterrows():
            g = gold_diagnoses(r)
            recs.append({"id": int(r["stay_id"]), "gold": (g["primary"] + g["secondary"]),
                         "sents": split_sentences(case_presentation(r)), "labs": {}})
    elif cohort == "cardiac_mm":
        from provena_med.core.cohort_mm import load_mm_cases
        for c in load_mm_cases(n, seed):
            recs.append({"id": int(c["id"]), "gold": c["gold"],
                         "sents": split_sentences(c.get("HPI", "")) + split_sentences(c.get("physical_exam", "")),
                         "labs": {}})
    elif cohort == "icu_mm":
        from provena_med.core.cohort_icu import load_icu_cases
        for c in load_icu_cases(n, seed):
            recs.append({"id": int(c["id"]), "gold": c["gold"],
                         "sents": split_sentences(c.get("HPI", "")) + split_sentences(c.get("physical_exam", "")),
                         "labs": labflat(c.get("labs"))})
    elif cohort == "eicu":
        from provena_med.core.cohort_eicu import load_eicu_cases
        for c in load_eicu_cases(n, seed):
            sents = [f"Past history: {s}" for s in c.get("past_history", [])] + \
                    [f"Exam: {s}" for s in c.get("physical_exam", [])]
            recs.append({"id": int(c["id"]), "gold": c["gold"], "sents": sents,
                         "labs": labflat(c.get("labs"))})
    elif cohort == "mimic3":
        from provena_med.core.cohort_mimic3 import load_mimic3_cases
        for c in load_mimic3_cases(n, seed):
            recs.append({"id": int(c["id"]), "gold": c["gold"],
                         "sents": split_sentences(c.get("HPI", "")) + split_sentences(c.get("physical_exam", "")),
                         "labs": labflat(c.get("labs"))})
    else:
        raise ValueError(cohort)
    return recs


def structured(label: str, labs: dict):
    """Return (applicable, met) for the objective lab signature of `label`."""
    low = label.lower()
    for pat, fn in DX_SIGNATURE:
        if re.search(pat, low):
            return True, bool(fn(labs))
    return False, False


def cohens_kappa(a: list[int], b: list[int]) -> float:
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = sum(((a == k).mean() * (b == k).mean()) for k in (0, 1))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


ALL_COHORTS = ["ed", "cardiac_mm", "icu_mm", "eicu", "mimic3"]


def triangulate_cohort(cohort, n, seed, max_labels, emb, judge, out=None) -> dict:
    recs = cohort_records(cohort, n, seed)
    rows = []  # per (case,label): emb_hit, judge_hit, applic, struct
    manifest = []
    for rec in recs:
        labels = [g for g in rec["gold"] if isinstance(g, str) and g.strip()][:max_labels]
        sents = [s for s in rec["sents"] if s.strip()][:40]
        case_labels = []
        sims = None
        judge_hits: list[float] = []
        if labels and sents:
            S = emb.encode(sents, normalize_embeddings=True, show_progress_bar=False)
            L = emb.encode(labels, normalize_embeddings=True, show_progress_bar=False)
            sims = L @ S.T  # (labels x sents)
            notes_txt = " ".join(sents)
            judge_hits = judge.supports_dx([notes_txt] * len(labels), labels)
        for li, lab in enumerate(labels):
            emb_hit = int(sims[li].max() >= TAU_EMB) if sims is not None else 0
            judge_hit = int(judge_hits[li]) if judge_hits else 0
            applic, met = structured(lab, rec["labs"])
            narr = int(emb_hit or judge_hit)
            level = 1 + narr + int(applic and met)  # CODE always 1
            rows.append((emb_hit, judge_hit, applic, int(applic and met)))
            case_labels.append({"label": lab, "level": level, "code": 1,
                                "narr_emb": emb_hit, "narr_judge": judge_hit,
                                "structured": (int(met) if applic else None)})
        manifest.append({"id": rec["id"], "labels": case_labels})

    emb_hits = [r[0] for r in rows]; judge_hits_all = [r[1] for r in rows]
    applic = [r[2] for r in rows]; struct_met = [r[3] for r in rows]
    narr = [int(a or b) for a, b in zip(emb_hits, judge_hits_all)]
    levels = [1 + n + s for n, s in zip(narr, struct_met)]
    N = max(1, len(rows))
    stats = {"cohort": cohort, "cases": len(recs), "labels": len(rows),
             "narrative": float(np.mean(narr)), "emb": float(np.mean(emb_hits)),
             "judge": float(np.mean(judge_hits_all)),
             "kappa_emb_judge": cohens_kappa(emb_hits, judge_hits_all),
             "structured_applicable": float(np.mean(applic)),
             "structured_met_of_applicable": float(np.sum(struct_met) / max(1, np.sum(applic))),
             "L1": float(np.mean([l == 1 for l in levels])), "L2": float(np.mean([l == 2 for l in levels])),
             "L3": float(np.mean([l == 3 for l in levels])),
             "triangulated": float(np.mean([l >= 2 for l in levels]))}
    print(f"\n=== triangulation: {cohort} | {len(recs)} cases | {len(rows)} gold labels | judge={judge.name} ===")
    print(f"  narrative corroboration (emb OR judge): {stats['narrative']:.3f}")
    print(f"     BioLORD-embedding extractor       : {stats['emb']:.3f}")
    print(f"     LLM-judge extractor               : {stats['judge']:.3f}")
    print(f"     extractor agreement (Cohen kappa) : {stats['kappa_emb_judge']:.3f}")
    print(f"  structured signature applicable      : {stats['structured_applicable']:.3f}")
    print(f"     of applicable, lab signature met  : {stats['structured_met_of_applicable']:.3f}")
    print(f"  corroboration level distribution     : L1={stats['L1']:.3f}  L2={stats['L2']:.3f}  L3={stats['L3']:.3f}")
    print(f"  TRIANGULATED (level>=2) gold fraction: {stats['triangulated']:.3f}")

    out = out or f"outputs/triang_{cohort}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"  manifest -> {out}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=ALL_COHORTS + ["all"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-labels", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from provena_med.core.llm_judge import LLMJudge
    emb = SentenceTransformer("FremyCompany/BioLORD-2023")
    judge = LLMJudge()

    cohorts = ALL_COHORTS if args.cohort == "all" else [args.cohort]
    summary = []
    for ch in cohorts:
        out = args.out if (args.out and args.cohort != "all") else None
        summary.append(triangulate_cohort(ch, args.n, args.seed, args.max_labels, emb, judge, out))

    if len(summary) > 1:
        print(f"\n{'cohort':12}{'labels':>7}{'narr':>7}{'emb':>6}{'judge':>7}{'kappa':>7}{'triang':>8}")
        print("-" * 54)
        for s in summary:
            print(f"{s['cohort']:12}{s['labels']:>7}{s['narrative']:>7.3f}{s['emb']:>6.3f}"
                  f"{s['judge']:>7.3f}{s['kappa_emb_judge']:>7.3f}{s['triangulated']:>8.3f}")


if __name__ == "__main__":
    main()
