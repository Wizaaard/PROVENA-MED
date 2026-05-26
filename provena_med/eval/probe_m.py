"""(M) attribution probe (PoC): does the model's computation actually use a cited unit?

Operationalizes activation patching as ATTENTION KNOCKOUT: we re-score the model's own
committed claim, P(claim | prompt), then make a cited unit e_i invisible to attention
(attention_mask=0 at e_i's token positions, with explicit position_ids so the sequence
stays aligned) and re-score. The drop  Delta^M_i = logP_factual - logP_knockout  measures
whether the computation routed through e_i. Control: knock out a random *un*cited unit
(should give ~0). Open-weight only; text cited units; small sample.

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python probe_m.py --n 6 --model .../Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.mmais import extract_claims_lenient  # noqa: E402
from provena_med.tasks.run_generate_staged import build_messages, load_provena  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--cohort", default="cardiac_mm")
    ap.add_argument("--staged", default="outputs/staged_prov_cardiac_mm_llama31_8b.jsonl")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-claims", type=int, default=6)
    ap.add_argument("--out", default=None, help="save per-citation rows (claim/unit text + Delta) as JSONL")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    dev = model.device

    def render(header, bundle):
        msgs = build_messages(header, bundle)
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # templates that forbid a separate system role (e.g. BioMistral)
            merged, sys_txt = [], ""
            for m in msgs:
                if m["role"] == "system":
                    sys_txt = m["content"]
                elif m["role"] == "user":
                    merged.append({"role": "user", "content": (sys_txt + "\n\n" + m["content"]).strip()})
                    sys_txt = ""
                else:
                    merged.append(m)
            return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def claim_logprob(prompt_ids, claim_ids, knockout_pos):
        ids = torch.tensor([prompt_ids + claim_ids], device=dev)
        attn = torch.ones_like(ids)
        for p in knockout_pos:
            attn[0, p] = 0
        pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        logits = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits[0]
        start = len(prompt_ids)
        lp = torch.log_softmax(logits[start - 1:-1].float(), -1)
        tgt = ids[0, start:]
        return lp[range(len(tgt)), tgt].mean().item()

    # header+bundle per case (matches generation); staged output per case
    cases = {c["id"]: c for c in load_provena(args.cohort, 200, 0)}
    staged = {r["stay_id"]: r for r in (json.loads(l) for l in open(args.staged))}
    ids_common = [i for i in staged if i in cases][:args.n]

    rows = []
    for cid in ids_common:
        case, out = cases[cid], staged[cid]
        prompt = render(case["header"], case["bundle"])
        enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        pids, offs = enc["input_ids"], enc["offset_mapping"]
        unit_ids = [u["id"] for u in case["bundle"]]
        # char span of each unit's "[id] ..." line in the prompt
        span = {}
        for uid in unit_ids:
            k = prompt.find(f"[{uid}]")
            if k >= 0:
                span[uid] = (k, prompt.find("\n", k) if prompt.find("\n", k) > 0 else len(prompt))
        umap = {u["id"]: u for u in case["bundle"]}
        for c in extract_claims_lenient(out["output"])[:args.max_claims]:
            cited = [e for e in c["evidence"] if e in umap and e in span]
            if not cited:
                continue
            claim_ids = tok(" " + c["claim"], add_special_tokens=False)["input_ids"]
            base = claim_logprob(pids, claim_ids, [])
            for e in cited:
                lo, hi = span[e]
                ko = [i for i, (a, b) in enumerate(offs) if a >= lo and a < hi]
                d = base - claim_logprob(pids, claim_ids, ko)
                rows.append({"kind": "cited", "id": cid, "claim": c["claim"], "unit": e,
                             "unit_text": umap[e]["text"], "mod": umap[e]["type"], "delta": d})
            # control: a random uncited unit present in the prompt
            ctrl_u = [u for u in span if u not in cited]
            if ctrl_u:
                e = random.Random(cid).choice(ctrl_u)
                lo, hi = span[e]
                ko = [i for i, (a, b) in enumerate(offs) if a >= lo and a < hi]
                rows.append({"kind": "control", "id": cid, "claim": c["claim"], "unit": e,
                             "unit_text": umap[e]["text"], "mod": umap[e]["type"],
                             "delta": base - claim_logprob(pids, claim_ids, ko)})

    import numpy as np
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("wrote", len(rows), "rows ->", args.out)
    cited = [r["delta"] for r in rows if r["kind"] == "cited"]
    ctrl = [r["delta"] for r in rows if r["kind"] == "control"]
    print(f"\n=== (M) attention-knockout probe | {args.model.split('/')[-1]} | {len(cited)} cited, {len(ctrl)} control ===")
    print(f"  mean Delta^M  cited   = {np.mean(cited):+.3f}  (drop in claim logprob when the cited unit is knocked out)")
    print(f"  mean Delta^M  control = {np.mean(ctrl):+.3f}  (random uncited unit)")
    print(f"  fraction of cited with Delta^M>0 (model used it): {np.mean([d>0 for d in cited]):.2f}")
    print(f"  fraction of cited with Delta^M>0.05            : {np.mean([d>0.05 for d in cited]):.2f}")
    by = {}
    for r in rows:
        if r["kind"] == "cited":
            by.setdefault(r["mod"], []).append(r["delta"])
    for m, ds in sorted(by.items()):
        print(f"     {m:14} mean Delta^M = {np.mean(ds):+.3f} (n={len(ds)})")


if __name__ == "__main__":
    main()
