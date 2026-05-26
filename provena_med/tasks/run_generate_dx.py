"""PROVENA-MED diagnosis track: full-information ranked-differential generation for any cohort.

Unlike the staged artifact (whose `differential` field some models under-populate), this uses a
focused prompt that reliably elicits {"differential": [...]} from every model, so Hit@k is a fair
cross-model accuracy measure. Full information (the whole evidence bundle is shown). Score with
score_diagnosis.py (BioLORD match to the gold primary diagnosis).

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_generate_dx.py \
      --model .../gemma-3-27b-it --cohort cardiac_mm --n 150 --out outputs/dx_cardiac_mm_gemma3_27b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import bundle_to_prompt  # noqa: E402
from provena_med.tasks.run_generate_staged import load_provena  # noqa: E402

PERSONA = "You are an experienced physician producing a ranked differential diagnosis."


def build_messages(header: str, units: list[dict]) -> list[dict]:
    user = (
        f"{header}\n\n"
        "PATIENT RECORD:\n"
        f"{bundle_to_prompt(units)}\n\n"
        "Based only on the record above, list the most likely diagnoses, most likely first. "
        'Return ONLY a JSON object of the form {"differential": ["diagnosis 1", "diagnosis 2", ...]} '
        "with up to 10 concise diagnosis names and no other text."
    )
    return [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]


def render(tok, header, units):
    msgs = build_messages(header, units)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = load_provena(args.cohort, args.n, args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    prompts = [render(tok, c["header"], c["bundle"]) for c in cases]
    outputs: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
        print(f"[dx] {min(start + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for c, text in zip(cases, outputs):
            f.write(json.dumps({"stay_id": c["id"], "gold": c["gold"], "output": text}) + "\n")
    print(f"[dx] model={args.model} cohort={args.cohort} -> {out_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
