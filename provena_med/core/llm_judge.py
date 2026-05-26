"""LLM-as-Judge verifier for PROVENA-MED final scoring (replaces the weak NLI proxy).

A strong, held-out instruction model judges support. Exposes the SAME `.entail(premises,
hypotheses) -> list[float]` interface as mmais.NLIVerifier, so it is a drop-in:
    score_records(records, verifier=LLMJudge())
returns 1.0 when the evidence supports the claim, else 0.0. Also `.supports_dx(notes,
diagnosis)` for the triangulation narrative check.

Default judge = Llama-3.3-70B-Instruct: strong, and NOT in the evaluated model panel, so it
cannot self-prefer. NLI stays available as a fast dev proxy; final numbers use this judge.
"""
from __future__ import annotations

import re
from pathlib import Path

DEFAULT_JUDGE = "~/models/Llama-3.3-70B-Instruct"

_SUPPORT_SYS = "You are a meticulous clinical evidence auditor. Judge only from the evidence given."
_SUPPORT_USER = (
    "EVIDENCE:\n{premise}\n\nCLAIM:\n{hypothesis}\n\n"
    "Does the EVIDENCE directly support the CLAIM? A supporting evidence item states or "
    "clearly implies the claim. Answer with exactly one word: SUPPORTS or UNSUPPORTED."
)
_DX_SYS = "You are a careful clinical chart reviewer."
_DX_USER = (
    "CLINICAL NOTES:\n{notes}\n\n"
    "Do these notes indicate the patient has, or is being treated for, \"{dx}\"? "
    "Answer with exactly one word: YES or NO."
)


class LLMJudge:
    def __init__(self, model_name: str = DEFAULT_JUDGE, max_new_tokens: int = 4,
                 batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = Path(model_name).name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def _render(self, sys_txt: str, user_txt: str) -> str:
        msgs = [{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": sys_txt + "\n\n" + user_txt}],
                tokenize=False, add_generation_prompt=True)

    def _run(self, prompts: list[str], positive: str) -> list[float]:
        # left-boundary match so "unsupported" does NOT match positive="support"
        pat = re.compile(r"(?<![a-z])" + re.escape(positive))
        out: list[float] = []
        for s in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[s:s + self.batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=False, truncation=True, max_length=2048).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          do_sample=False, pad_token_id=self.tok.pad_token_id)
            dec = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.extend(1.0 if pat.search(d.strip().lower()) else 0.0 for d in dec)
        return out

    def entail(self, premises: list[str], hypotheses: list[str], bs: int | None = None) -> list[float]:
        prompts = [self._render(_SUPPORT_SYS, _SUPPORT_USER.format(premise=p[:1200], hypothesis=h[:600]))
                   for p, h in zip(premises, hypotheses)]
        return self._run(prompts, "support")

    def supports_dx(self, notes: list[str], diagnoses: list[str]) -> list[float]:
        prompts = [self._render(_DX_SYS, _DX_USER.format(notes=n[:3000], dx=d))
                   for n, d in zip(notes, diagnoses)]
        return self._run(prompts, "yes")
