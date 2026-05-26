"""Cross W (judge support) x M (attention-knockout use) -> the MM-AIS quadrants.

Reads the per-citation M rows from probe_m.py, asks the held-out judge whether each cited
unit supports its claim (W, the singleton-sufficiency case), thresholds M at tau_M, and
tabulates the four quadrants: true provenance (W+M+), post-hoc rationalization (W+M-),
misgrounded reliance (W-M+), decorative (W-M-).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.llm_judge import LLMJudge

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="outputs/m_probe_llama31.jsonl")
ap.add_argument("--tau-m", type=float, default=0.05)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.inp) if json.loads(l)["kind"] == "cited"]
judge = LLMJudge()
W = judge.entail([r["unit_text"] for r in rows], [r["claim"] for r in rows])
q = {"W+M+": 0, "W+M-": 0, "W-M+": 0, "W-M-": 0}
for r, w in zip(rows, W):
    m = r["delta"] > args.tau_m
    q[f"W{'+' if w >= 1 else '-'}M{'+' if m else '-'}"] += 1
n = max(1, len(rows))
print(f"\n=== MM-AIS W x M quadrants | {Path(args.inp).stem} | {len(rows)} citations | tau_M={args.tau_m} ===")
print(f"{'':18}{'M+ (used)':>14}{'M- (unused)':>14}")
print(f"{'W+ (supports)':18}{q['W+M+']:>10} ({100*q['W+M+']/n:.0f}%){q['W+M-']:>9} ({100*q['W+M-']/n:.0f}%)")
print(f"{'W- (no support)':18}{q['W-M+']:>10} ({100*q['W-M+']/n:.0f}%){q['W-M-']:>9} ({100*q['W-M-']/n:.0f}%)")
print(f"\n  true provenance (W+M+)        : {100*q['W+M+']/n:.0f}%")
print(f"  post-hoc rationalization(W+M-): {100*q['W+M-']/n:.0f}%")
print(f"  misgrounded reliance (W-M+)   : {100*q['W-M+']/n:.0f}%  (unsafe)")
print(f"  decorative (W-M-)             : {100*q['W-M-']/n:.0f}%")
with open("outputs/quadrants_llama31.json", "w") as f:
    json.dump({"n": len(rows), "tau_m": args.tau_m, "quadrants": q}, f, indent=2)
