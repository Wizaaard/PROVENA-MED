"""PROVENA-MED Experiment 0 — phase 1: differential generation with a local open model.

Full-information ED differential diagnosis. Text-only. Writes one JSONL record per case.
Uses HuggingFace transformers (the env's vLLM build is ABI-incompatible with torch 2.10).

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_generate.py \
      --n 30 --out outputs/exp0_llama31_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core import edcds  # noqa: E402

DEFAULT_MODEL = "~/models/Llama-3.1-8B-Instruct"


def build_messages(row) -> list[dict]:
    presentation = edcds.case_presentation(row)
    system = (
        "You are an experienced emergency medicine physician. Based only on the "
        "initial presentation provided, produce a ranked differential diagnosis."
    )
    user = (
        f"{presentation}\n\n"
        "List the most likely diagnoses, most likely first. Return ONLY a JSON "
        'object of the form {"differential": ["diagnosis 1", "diagnosis 2", ...]} '
        "with up to 10 concise diagnosis names and no other text."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = edcds.load_cases(n=args.n, seed=args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    prompts = [
        tok.apply_chat_template(
            build_messages(row), tokenize=False, add_generation_prompt=True
        )
        for _, row in df.iterrows()
    ]

    outputs: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tok.batch_decode(new, skip_special_tokens=True))
        print(f"[generate] {min(start + args.batch_size, len(prompts))}/{len(prompts)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for (_, row), text in zip(df.iterrows(), outputs):
            rec = {
                "stay_id": int(row["stay_id"]),
                "gold": edcds.gold_diagnoses(row),
                "output": text,
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[generate] model={args.model}")
    print(f"[generate] wrote {len(df)} records -> {out_path}")


if __name__ == "__main__":
    main()
