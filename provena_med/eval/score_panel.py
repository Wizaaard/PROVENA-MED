"""Phase 2 of the panel: score every staged_prov_<cohort>_<model>.jsonl with one judge load.

Reports, per (model, cohort): parse rate, citation validity, MM-AIS precision (LLM-judge,
overall + by modality) and salient-evidence recall (objective, overall + by modality).
This is the headline leaderboard table.

  CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python score_panel.py --cohort cardiac_mm
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from provena_med.core.llm_judge import LLMJudge
from provena_med.core.mmais import score_records, score_recall

MODS = ["NOTE_SPAN", "TABLE_ROW", "IMAGE_FINDING", "GUIDELINE_RULE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="*", help="cohort filter, or comma-list (e.g. ed,icu_mm)")
    ap.add_argument("--glob", default=None, help="override input glob")
    ap.add_argument("--out", default="outputs/panel_scores.jsonl")
    args = ap.parse_args()

    if args.glob:
        files = sorted(glob.glob(args.glob))
    else:
        files = sorted({f for ch in args.cohort.split(",")
                        for f in glob.glob(f"outputs/staged_prov_{ch}_*.jsonl")})
    if not files:
        pattern = args.glob or f"outputs/staged_prov_{args.cohort}_*.jsonl"
        print(f"no files match {pattern}")
        return
    judge = LLMJudge()
    rows = []
    hdr = (f"{'cohort':11}{'model':13}{'parsed':>7}{'valid':>6}{'MM-AIS':>7}"
           f"{'NOTE':>6}{'TABL':>6}{'IMG':>6}{'GUID':>6} | {'recall':>6}{'r.lab':>6}{'r.img':>6}{'r.gid':>6}")
    print(hdr); print("-" * len(hdr))
    COHORT_NAMES = ["cardiac_mm", "icu_mm", "mimic3", "eicu", "ed"]  # match longest-first
    for f in files:
        rest = Path(f).stem[len("staged_prov_"):]  # <cohort>_<model> (model has underscores)
        cohort, model = "?", rest
        for ch in COHORT_NAMES:
            if rest.startswith(ch + "_"):
                cohort, model = ch, rest[len(ch) + 1:]
                break
        recs = [json.loads(l) for l in open(f)]
        p = score_records(recs, verifier=judge)
        rc = score_recall(recs)
        bm, rbm = p["mm_ais_precision_by_modality"], rc["recall_by_modality"]
        rows.append({"cohort": cohort, "model": model, **p, "recall": rc["recall"],
                     "recall_by_modality": rbm, "n_salient": rc["n_salient"]})
        g = lambda d, k: d.get(k, float("nan"))
        print(f"{cohort:11}{model:13}{p['parsed']:>4}/{p['n']:<2}{p['citation_validity']:>6.2f}"
              f"{p['mm_ais_precision']:>7.2f}{g(bm,'NOTE_SPAN'):>6.2f}{g(bm,'TABLE_ROW'):>6.2f}"
              f"{g(bm,'IMAGE_FINDING'):>6.2f}{g(bm,'GUIDELINE_RULE'):>6.2f} | {rc['recall']:>6.2f}"
              f"{g(rbm,'TABLE_ROW'):>6.2f}{g(rbm,'IMAGE_FINDING'):>6.2f}{g(rbm,'GUIDELINE_RULE'):>6.2f}")
    with open(args.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
