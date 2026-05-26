"""Judge-phase panel: load the held-out LLM judge ONCE, then for every model compute
both (W) causal MM-AIS (Shapley / necessity / sufficiency from the staged outputs) and
the W x M quadrants (from the per-citation rows written by probe_m.py). Loading the 70B
judge is the expensive step, so we amortize it over the whole model slate.

  CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python panel_judge.py \
      --cohort cardiac_mm --ids gemma3_12b gemma3_27b ... --n 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.eval.causal_mmais import score_causal_W  # noqa: E402
from provena_med.core.llm_judge import LLMJudge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cardiac_mm")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tau-m", type=float, default=0.05)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--causal-out", default="outputs/causalW_panel.jsonl")
    ap.add_argument("--quad-out", default="outputs/quadrants_panel.jsonl")
    ap.add_argument("--skip-causal", action="store_true", help="quadrants only (re-run for missing M files)")
    args = ap.parse_args()

    judge = LLMJudge()
    cf = open(args.causal_out, "a")
    qf = open(args.quad_out, "a")
    for mid in args.ids:
        # ---- (W) causal MM-AIS from the staged cited generation ----
        staged = f"outputs/staged_prov_{args.cohort}_{mid}.jsonl"
        if not args.skip_causal and Path(staged).exists():
            recs = [json.loads(l) for l in open(staged)][:args.n]
            r = score_causal_W(recs, judge, tau=args.tau, max_k=args.max_k)
            cf.write(json.dumps({"model": mid, "cohort": args.cohort, **r}) + "\n")
            cf.flush()
            print(f"[causal] {mid}: claims_scored={r['claims_scored']}", flush=True)
        else:
            print(f"[causal] skip missing {staged}", flush=True)

        # ---- W x M quadrants from the attention-knockout rows ----
        mfile = f"outputs/m_probe_{args.cohort}_{mid}.jsonl"
        if Path(mfile).exists():
            rows = [json.loads(l) for l in open(mfile) if json.loads(l)["kind"] == "cited"]
            if rows:
                W = judge.entail([x["unit_text"] for x in rows], [x["claim"] for x in rows])
                q = {"W+M+": 0, "W+M-": 0, "W-M+": 0, "W-M-": 0}
                for x, w in zip(rows, W):
                    m = x["delta"] > args.tau_m
                    q[f"W{'+' if w >= 1 else '-'}M{'+' if m else '-'}"] += 1
                qf.write(json.dumps({"model": mid, "cohort": args.cohort, "n": len(rows),
                                     "tau_m": args.tau_m, "quadrants": q}) + "\n")
                qf.flush()
                print(f"[quad] {mid}: n={len(rows)} {q}", flush=True)
            else:
                print(f"[quad] {mid}: no cited rows", flush=True)
        else:
            print(f"[quad] skip missing {mfile}", flush=True)
    cf.close()
    qf.close()
    print("[panel_judge] DONE", flush=True)


if __name__ == "__main__":
    main()
