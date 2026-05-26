"""PROVENA-MED interactive track — scorer (with bootstrap CIs + evidence efficiency).

oracle: diagnostic-yield curve (Hit@k with 95% CI per reveal condition).
agent:  request behavior, final Hit@k (CI), and an Evidence-Efficiency Score
        EES = Hit@5 - lambda * (mean_requests / budget).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from provena_med.core.dxmatch import bootstrap_ci, hit_recall

EMBED_MODEL = "FremyCompany/BioLORD-2023"
COND_ORDER = ["chief_only", "history", "history_tests", "all"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "agent"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.5, help="efficiency penalty weight")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    records = [json.loads(l) for l in open(args.inp)]
    embedder = SentenceTransformer(EMBED_MODEL)
    sk = 5 if 5 in args.ks else max(args.ks)

    if args.mode == "oracle":
        by_cond = defaultdict(list)
        for r in records:
            by_cond[r["condition"]].append((r.get("differential", []), r["gold"]["primary"]))
        n = len(next(iter(by_cond.values())))
        print(f"\n=== Interactive oracle yield curve ({args.inp}) | n={n} | tau={args.tau} ===")
        print("condition".ljust(15) + "".join(f"Hit@{k}".rjust(8) for k in args.ks)
              + f"   Hit@{sk} 95% CI")
        for cond in COND_ORDER:
            if cond not in by_cond:
                continue
            m = hit_recall(by_cond[cond], embedder, args.ks, args.tau)
            lo, hi = bootstrap_ci(m["hit_pc"][sk])
            row = cond.ljust(15) + "".join(f"{m['hit'][k]:.3f}".rjust(8) for k in args.ks)
            print(row + f"   [{lo:.3f}, {hi:.3f}]")
        return

    # ---- agent ----
    pairs = [(r.get("differential", []), r["gold"]["primary"]) for r in records]
    m = hit_recall(pairs, embedder, args.ks, args.tau)
    n = len(records)
    n_reqs = np.array([r["n_requests"] for r in records], dtype=float)
    mean_req = float(n_reqs.mean())
    early = np.array([1.0 if (r.get("committed") and r["n_requests"] < args.budget) else 0.0
                      for r in records])
    first_req = Counter(r["requests"][0] if r["requests"] else "<none>" for r in records)
    item_hist = Counter(it for r in records for it in r["requests"])

    hit5 = np.array(m["hit_pc"][sk])
    ees_pc = hit5 - args.lam * (n_reqs / args.budget)
    h_lo, h_hi = bootstrap_ci(hit5)
    e_lo, e_hi = bootstrap_ci(ees_pc)

    print(f"\n=== Interactive agent ({args.inp}) | n={n} | tau={args.tau} | budget={args.budget} ===")
    print("-- behavior --")
    print(f"  mean requests used : {mean_req:.2f} / {args.budget}")
    print(f"  early-commit rate  : {early.mean():.3f}  (committed before exhausting budget)")
    print(f"  request-count dist : {dict(sorted(Counter(int(x) for x in n_reqs).items()))}")
    print(f"  first request      : {dict(first_req)}")
    print(f"  items requested    : {dict(item_hist)}")
    print("-- accuracy --")
    for k in args.ks:
        print(f"  Hit@{k}={m['hit'][k]:.3f}   Recall@{k}={m['recall'][k]:.3f}")
    print(f"  Hit@{sk} 95% CI    : [{h_lo:.3f}, {h_hi:.3f}]")
    print("-- evidence efficiency --")
    print(f"  EES = Hit@{sk} - {args.lam}*(req/budget) = {ees_pc.mean():.3f}  95% CI [{e_lo:.3f}, {e_hi:.3f}]")
    print(f"  yield per request  : {(m['hit'][sk]/max(mean_req,1e-9)):.3f}  (Hit@{sk} per request)")
    strat = defaultdict(list)
    for r, (p, g) in zip(records, pairs):
        strat[r["n_requests"]].append((p, g))
    print(f"  Hit@{sk} by #requests:")
    for nr in sorted(strat):
        mm = hit_recall(strat[nr], embedder, [sk], args.tau)
        print(f"    {nr} req (n={len(strat[nr])}): {mm['hit'][sk]:.3f}")


if __name__ == "__main__":
    main()
