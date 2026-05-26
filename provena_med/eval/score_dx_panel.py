"""Score the diagnosis-track panel: Hit@1 / Hit@3 / Hit@5 against the cohort's
PRINCIPAL diagnosis (gold['primary'][0]) for every model x cohort dx file.

Why principal-only: the cohorts disagree on gold cardinality (ED has 1-2 primary
codes, the inpatient cohorts list up to ~18 billed ICDs). Matching top-k against
the principal diagnosis makes Hit@k apples-to-apples across cohorts. Matching is
BioLORD cosine at tau=0.75 (same as score_diagnosis.py); runs CPU-only by default
so it does not contend with the dx generation panel for GPUs.

  conda run -n ragcon python score_dx_panel.py \
      --in 'outputs/dx_*.jsonl' --out outputs/dx_panel.jsonl --tau 0.75
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

EMBED_MODEL = "FremyCompany/BioLORD-2023"
_FNAME = re.compile(r"^dx_(?P<cohort>ed|cardiac_mm|icu_mm|eicu|mimic3)_(?P<model>.+)\.jsonl$")


def parse_diff(text: str) -> list[str]:
    if not text:
        return []
    s = text.strip().lstrip("`").lstrip("json").strip()
    # outermost JSON object
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j >= 0:
        try:
            obj = json.loads(s[i:j + 1])
            d = obj.get("differential") if isinstance(obj, dict) else None
            if isinstance(d, list):
                return [str(x).strip() for x in d if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # fallback: a bare JSON array
    m = re.search(r"\[.*?\]", s, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def principal_gold(rec) -> str | None:
    g = rec.get("gold", {}).get("primary")
    if isinstance(g, list):
        return g[0] if g else None
    return g if isinstance(g, str) and g.strip() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/dx_*.jsonl", help="glob over dx files")
    ap.add_argument("--out", default="outputs/dx_panel.jsonl")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    files = sorted(glob.glob(args.glob))
    rows = []  # (cohort, model, file, [(preds, gold, idx)])
    all_terms: list[str] = []
    term_index: list[tuple] = []  # (row_i, kind, pos_or_None)
    for f in files:
        m = _FNAME.match(Path(f).name)
        if not m:
            continue
        cohort, model = m["cohort"], m["model"]
        recs = [json.loads(l) for l in open(f)]
        cases = []  # list of (preds, gold_principal)
        for r in recs:
            preds = parse_diff(r["output"])
            g = principal_gold(r)
            if not preds or not g:
                cases.append((preds, g))
                continue
            cases.append((preds, g))
        ri = len(rows)
        rows.append({"cohort": cohort, "model": model, "file": f, "cases": cases})
        for pi, (preds, g) in enumerate(cases):
            for k_, p in enumerate(preds):
                term_index.append((ri, pi, "pred", k_))
                all_terms.append(p)
            if g:
                term_index.append((ri, pi, "gold", 0))
                all_terms.append(g)

    if not all_terms:
        print("No predictions/gold parsed from any dx file under", args.glob)
        return
    print(f"[score-dx] loading BioLORD (device={args.device}) for {len(all_terms)} terms across "
          f"{len(rows)} (cohort,model) files...", flush=True)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL, device=args.device)
    emb = embedder.encode(all_terms, normalize_embeddings=True, show_progress_bar=False,
                          batch_size=128)

    # repackage into rows[ri].cases[pi].{pred_emb[list], gold_emb}
    by_case: dict = {}
    for (ri, pi, kind, k_), e in zip(term_index, emb):
        slot = by_case.setdefault((ri, pi), {"pred": {}, "gold": None})
        if kind == "pred":
            slot["pred"][k_] = e
        else:
            slot["gold"] = e

    out_rows = []
    for ri, row in enumerate(rows):
        n = len(row["cases"])
        hit = {k: 0 for k in args.ks}
        parsed = 0
        for pi, (preds, g) in enumerate(row["cases"]):
            if not preds or g is None:
                continue
            parsed += 1
            slot = by_case[(ri, pi)]
            gv = slot["gold"]
            if gv is None:
                continue
            for k in args.ks:
                top = [slot["pred"][i] for i in range(min(k, len(preds)))]
                if not top:
                    continue
                sims = np.stack(top) @ gv
                if sims.max() >= args.tau:
                    hit[k] += 1
        rec = {"cohort": row["cohort"], "model": row["model"], "n": n, "parsed": parsed,
               **{f"hit@{k}": hit[k] / max(1, n) for k in args.ks}}
        out_rows.append(rec)
        print(f"  {row['model']:16} {row['cohort']:11} "
              f"parsed={parsed:3}/{n} "
              + "  ".join(f"H@{k}={hit[k] / max(1, n):.3f}" for k in args.ks), flush=True)

    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # panel summary: mean over cohorts per model
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in out_rows:
        by_model[r["model"]].append(r)
    print("\n=== panel mean (over cohorts) ===")
    print(f"{'model':16}  " + "  ".join(f"H@{k}".rjust(6) for k in args.ks))
    for m, rs in sorted(by_model.items(), key=lambda kv: -np.mean([r["hit@1"] for r in kv[1]])):
        means = {k: np.mean([r[f"hit@{k}"] for r in rs]) for k in args.ks}
        print(f"{m:16}  " + "  ".join(f"{means[k]:.3f}".rjust(6) for k in args.ks))
    print(f"\n[score-dx] wrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
