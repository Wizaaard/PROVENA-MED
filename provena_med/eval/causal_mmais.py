"""(W) Causal attribution layer for MM-AIS: provenance as intervention, not correlation.

The judge is the set-support function S(c,A) = does the evidence SUBSET A support claim c.
On top of it we compute, per cited evidence unit e_i of claim c:
  * Shapley credit  phi^W_i  -- e_i's marginal support averaged over coalitions of the other
                              cited units (so redundant/overdetermined evidence shares credit,
                              unlike the singleton precision metric). SOUND iff phi^W_i > tau.
  * necessity       do(e_i <- empty): S(E_c)=1 and S(E_c\{e_i})=0  (claim needs e_i)
  * sufficiency     S({e_i})=1                                       (e_i alone supports c)
Reported modality-stratified. Exact Shapley for claims citing <= max_k units (covers the
vast majority); larger claims are skipped from the Shapley estimate (still get nec/suff).

  CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 python causal_mmais.py \
      --in outputs/staged_prov_cardiac_mm_gemma3_12b.jsonl --n 40
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations

from provena_med.core.mmais import extract_claims_lenient


def shapley(units: list[str], v: dict) -> dict:
    """units = cited ids; v = {frozenset(ids): 0/1 support}. Exact Shapley over coalitions."""
    k = len(units)
    phi = {u: 0.0 for u in units}
    for u in units:
        rest = [x for x in units if x != u]
        for r in range(len(rest) + 1):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for T in combinations(rest, r):
                phi[u] += w * (v[frozenset(T) | {u}] - v[frozenset(T)])
    return phi


def score_causal_W(records, judge, tau: float = 0.5, max_k: int = 4, max_claims: int | None = None):
    # ---- pass 1: gather every (claim, coalition) whose support we need ----
    tasks = []   # (rec_i, claim_i, units[list of ids], unit_text{id->text}, unit_mod{id->mod})
    need = {}    # (claim_uid, frozenset(ids)) -> index into judge batch
    prem, hyp = [], []
    n_claims = 0
    for ri, rec in enumerate(records):
        umap = {u["id"]: u for u in rec["bundle"]}
        for ci, c in enumerate(extract_claims_lenient(rec["output"])):
            cited = [e for e in c["evidence"] if e in umap]
            cited = list(dict.fromkeys(cited))           # dedup, keep order
            if not (1 <= len(cited) <= max_k):
                continue
            n_claims += 1
            if max_claims and n_claims > max_claims:
                break
            cu = f"{ri}:{ci}"
            texts = {e: umap[e]["text"] for e in cited}
            mods = {e: umap[e]["type"] for e in cited}
            tasks.append((cu, c["claim"], cited, texts, mods))
            for r in range(len(cited) + 1):              # all subsets (incl. empty)
                for T in combinations(cited, r):
                    key = (cu, frozenset(T))
                    if key not in need:
                        if not T:
                            need[key] = -1               # v(empty)=0, no judge call
                        else:
                            need[key] = len(prem)
                            prem.append("\n".join(texts[e][:600] for e in T))
                            hyp.append(c["claim"])
    # ---- batch-judge all non-empty coalitions ----
    ent = judge.entail(prem, hyp) if prem else []
    def v_of(cu, ids):
        idx = need[(cu, frozenset(ids))]
        return 0 if idx == -1 else int(ent[idx])

    # ---- pass 2: Shapley / necessity / sufficiency per cited unit ----
    agg = {}  # mod -> dict of counts
    def bump(mod, **kw):
        d = agg.setdefault(mod, {"n": 0, "sound": 0, "nec": 0, "suff": 0, "phi": 0.0})
        d["n"] += 1
        for k_, val in kw.items():
            d[k_] += val
    for cu, claim, cited, texts, mods in tasks:
        v = {frozenset(T): v_of(cu, T) for r in range(len(cited) + 1) for T in combinations(cited, r)}
        phi = shapley(cited, v)
        full = frozenset(cited)
        for e in cited:
            nec = int(v[full] == 1 and v[full - {e}] == 0)
            suff = int(v[frozenset({e})] == 1)
            bump(mods[e], sound=int(phi[e] > tau), nec=nec, suff=suff, phi=phi[e])
    return {"claims_scored": len(tasks), "by_modality": agg, "tau": tau, "max_k": max_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--out", default=None, help="append the result dict (with file/model) as JSONL")
    args = ap.parse_args()
    from pathlib import Path
    from provena_med.core.llm_judge import LLMJudge
    recs = [json.loads(l) for l in open(args.inp)][:args.n]
    r = score_causal_W(recs, LLMJudge(), tau=args.tau, max_k=args.max_k)
    print(f"\n=== (W) causal MM-AIS: {args.inp} | claims scored {r['claims_scored']} (cited<= {args.max_k}) ===")
    print(f"{'modality':16}{'n':>5}{'Shapley-sound':>14}{'necessity':>11}{'sufficiency':>13}{'mean phi':>10}")
    for mod, d in sorted(r["by_modality"].items()):
        n = max(1, d["n"])
        print(f"{mod:16}{d['n']:>5}{d['sound']/n:>14.3f}{d['nec']/n:>11.3f}{d['suff']/n:>13.3f}{d['phi']/n:>10.3f}")
    if args.out:
        rec = {"file": Path(args.inp).name, **r}
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[causal-W] appended -> {args.out}")


if __name__ == "__main__":
    main()
