"""PROVENA-MED Experiment 0 — phase 2: score differentials against gold (no GPU contention).

Parses each model's JSON differential, matches predicted diagnoses to gold primary
diagnoses by BioLORD embedding cosine similarity, and reports Hit@k / recall@k.
Run after generation so vLLM has released the GPU.

Example:
  HF_HUB_OFFLINE=1 conda run -n ragcon --no-capture-output \
    python score_diagnosis.py --in outputs/exp0_llama31_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

EMBED_MODEL = "FremyCompany/BioLORD-2023"


def parse_differential(text: str) -> list[str]:
    """Extract the ranked diagnosis list from a model response (robust to extra text)."""
    if not text:
        return []
    # Prefer an explicit JSON object containing "differential"
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "differential" in obj:
            items = obj["differential"]
            if isinstance(items, list):
                return [str(x).strip() for x in items if str(x).strip()]
    # Fallback: a bare JSON array
    m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--tau", type=float, default=0.75, help="cosine match threshold")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    records = [json.loads(line) for line in open(args.inp)]
    embedder = SentenceTransformer(EMBED_MODEL)

    # Collect all strings to embed once
    all_terms: list[str] = []
    index: list[tuple] = []  # (record_idx, kind, position)
    for ri, rec in enumerate(records):
        preds = parse_differential(rec["output"])
        rec["_preds"] = preds
        for pi, p in enumerate(preds):
            index.append((ri, "pred", pi))
            all_terms.append(p)
        for gi, g in enumerate(rec["gold"]["primary"]):
            index.append((ri, "gold", gi))
            all_terms.append(g)

    if not all_terms:
        print("No parseable predictions/gold found.")
        return
    emb = embedder.encode(all_terms, normalize_embeddings=True, show_progress_bar=False)
    vec = {}
    for (ri, kind, pos), e in zip(index, emb):
        vec.setdefault(ri, {"pred": {}, "gold": {}})[kind][pos] = e

    n = len(records)
    parsed_ok = sum(1 for r in records if r["_preds"])
    max_k = max(args.ks)
    hit_at = {k: 0 for k in args.ks}
    recall_at = {k: [] for k in args.ks}

    for ri, rec in enumerate(records):
        preds = rec["_preds"]
        gold = rec["gold"]["primary"]
        if not preds or not gold:
            for k in args.ks:
                recall_at[k].append(0.0)
            continue
        pv = vec[ri]["pred"]
        gv = vec[ri]["gold"]
        for k in args.ks:
            top = [pv[i] for i in range(min(k, len(preds)))]
            if not top:
                recall_at[k].append(0.0)
                continue
            top = np.stack(top)
            matched = 0
            case_hit = False
            for gi in range(len(gold)):
                sims = top @ gv[gi]
                if sims.max() >= args.tau:
                    matched += 1
                    case_hit = True
            if case_hit:
                hit_at[k] += 1
            recall_at[k].append(matched / len(gold))

    print(f"\n=== Experiment 0: ED differential ({args.inp}) ===")
    print(f"cases={n} | parseable_predictions={parsed_ok}/{n} | match tau={args.tau}")
    for k in args.ks:
        print(
            f"  Hit@{k} (>=1 gold primary matched): {hit_at[k]/n:.3f}"
            f"   |   Recall@{k} (gold primary covered): {np.mean(recall_at[k]):.3f}"
        )


if __name__ == "__main__":
    main()
