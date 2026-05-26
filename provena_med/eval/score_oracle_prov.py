"""Provenance (W judge) scorer for the 4 oracle-staged JSONLs.

Loads the LLM-as-judge (Llama-3.3-70B) once and scores each
  outputs/staged_oracle_<cond>_<model>.jsonl
for citation validity, MM-AIS precision (= Prec), salient-evidence recall and
attribution yield (= valid x prec). Prints a compact table and saves a JSONL.

Example:
  CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
      python score_oracle_prov.py --model-id llama31_8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from provena_med.core.llm_judge import LLMJudge
from provena_med.core.mmais import score_recall, score_records

CONDITIONS = ["chief_only", "history", "history_tests", "all"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="e.g. llama31_8b")
    ap.add_argument("--in-dir", default="outputs")
    ap.add_argument("--out", default=None,
                    help="output JSONL (default: outputs/oracle_prov_<model_id>.jsonl)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = []
    for cond in CONDITIONS:
        p = in_dir / f"staged_oracle_{cond}_{args.model_id}.jsonl"
        if not p.exists():
            print(f"[warn] missing {p}; skipping")
            continue
        files.append((cond, p))
    if not files:
        raise SystemExit(f"no staged_oracle_*_{args.model_id}.jsonl files found in {in_dir}")

    judge = LLMJudge()

    out_path = Path(args.out or in_dir / f"oracle_prov_{args.model_id}.jsonl")
    rows = []
    hdr = (f"{'condition':16}{'parsed':>9}{'valid':>7}{'prec':>7}"
           f"{'recall':>8}{'yield':>7}")
    print(hdr); print("-" * len(hdr))
    for cond, path in files:
        recs = [json.loads(line) for line in open(path)]
        p = score_records(recs, verifier=judge)
        rc = score_recall(recs)
        v, pr = p["citation_validity"], p["mm_ais_precision"]
        rcv = rc["recall"]
        yld = v * pr
        print(f"{cond:16}{p['parsed']:>5}/{p['n']:<3}{v:>7.2f}{pr:>7.2f}{rcv:>8.2f}{yld:>7.2f}")
        rows.append({
            "condition": cond, "model_id": args.model_id,
            "n": p["n"], "parsed": p["parsed"],
            "valid": v, "prec": pr, "recall": rcv, "yield": yld,
            "n_salient": rc["n_salient"], "n_scored_citations": p["n_scored_citations"],
            "prec_by_modality": p["mm_ais_precision_by_modality"],
            "recall_by_modality": rc["recall_by_modality"],
        })

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
