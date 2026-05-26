"""Staged provenance generation under the 4 oracle reveal conditions (ED, n=300).

Mirrors `run_interactive.py --mode oracle` (fixed evidence reveal) but uses the
*staged* schema from `run_generate_staged.py` so each oracle condition produces
cited claims, judgeable by the W judge. Writes one JSONL per condition in the
same format as `staged_ed_<model>.jsonl`.

Bundle is filtered by unit-ID prefix so the model can only cite evidence that
the condition's reveal set would have shown:
  chief_only       -> TABLE_ROW:vital:*                  (vitals from s0)
  history          -> + NOTE_SPAN:hpi:*                  (HPI)
  history_tests    -> + TABLE_ROW:lab:*                  (labs / tests)
  all              -> + NOTE_SPAN:med:*                  (home medications)

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_oracle_staged.py \
      --model /storage/.../Llama-3.1-8B-Instruct --n 300 \
      --out-dir outputs --model-id llama31_8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.tasks.run_generate_staged import build_messages, load_cases  # noqa: E402

# Unit-ID prefix sets per reveal condition (cumulative).
ORACLE_FILTERS = {
    "chief_only":    {"TABLE_ROW:vital:"},
    "history":       {"TABLE_ROW:vital:", "NOTE_SPAN:hpi:"},
    "history_tests": {"TABLE_ROW:vital:", "NOTE_SPAN:hpi:", "TABLE_ROW:lab:"},
    "all":           {"TABLE_ROW:vital:", "NOTE_SPAN:hpi:", "TABLE_ROW:lab:", "NOTE_SPAN:med:"},
}


def filter_bundle(bundle: list[dict], prefixes: set[str]) -> list[dict]:
    return [u for u in bundle if any(u["id"].startswith(p) for p in prefixes)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-id", required=True,
                    help="short id used in output filename, e.g. llama31_8b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--conditions", default="chief_only,history,history_tests,all")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = load_cases("ed", args.n, args.seed)
    print(f"[oracle-staged] loaded {len(cases)} ED cases (seed={args.seed})")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def render(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            merged, sys_txt = [], ""
            for m in msgs:
                if m["role"] == "system":
                    sys_txt = m["content"]
                elif m["role"] == "user":
                    merged.append({"role": "user",
                                   "content": (sys_txt + "\n\n" + m["content"]).strip()})
                    sys_txt = ""
                else:
                    merged.append(m)
            return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions.split(","):
        cond = cond.strip()
        if cond not in ORACLE_FILTERS:
            raise ValueError(f"unknown oracle condition: {cond}")
        prefixes = ORACLE_FILTERS[cond]

        cond_cases = []
        prompts = []
        for c in cases:
            filtered = filter_bundle(c["bundle"], prefixes)
            if not filtered:
                continue  # skip cases with no citable evidence under this condition
            cond_cases.append({**c, "bundle_filtered": filtered})
            prompts.append(render(build_messages(c["header"], filtered)))

        outputs: list[str] = []
        for s in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[s:s + args.batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            outputs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True))
            print(f"[oracle-staged:{cond}] "
                  f"{min(s + args.batch_size, len(prompts))}/{len(prompts)}")

        out_path = out_dir / f"staged_oracle_{cond}_{args.model_id}.jsonl"
        with out_path.open("w") as f:
            for c, text in zip(cond_cases, outputs):
                f.write(json.dumps({"stay_id": c["id"], "gold": c["gold"],
                                    "bundle": c["bundle_filtered"], "output": text}) + "\n")
        print(f"[oracle-staged:{cond}] wrote {len(cond_cases)} -> {out_path}")


if __name__ == "__main__":
    main()
