"""PROVENA-MED interactive track — runner (transformers backend).

Modes:
  oracle  - fixed reveal conditions (chief_only, +history, +history_tests, +all),
            single-turn; gives the diagnostic-yield (information-value) curve.
  agent   - the model decides what to REQUEST under a budget, then commits;
            batched across cases (stepped turn-by-turn) for speed.

Oracle and agent use the SAME physician persona and the SAME final commit format
({"differential": [...]}), so they differ only in interactivity.

Example:
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_interactive.py \
      --mode oracle --n 300 --out outputs/int_oracle_llama31_8b.jsonl
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_interactive.py \
      --mode agent --n 300 --budget 3 --out outputs/int_agent_llama31_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core import edcds_interactive as ei  # noqa: E402
from provena_med.core.dxmatch import parse_differential  # noqa: E402

DEFAULT_MODEL = "~/models/Llama-3.1-8B-Instruct"
ITEM_TITLE = {
    "history": "History of present illness",
    "labs_and_tests": "Laboratory and test results",
    "medications": "Medications",
}
CONDITIONS = {
    "chief_only": [],
    "history": ["history"],
    "history_tests": ["history", "labs_and_tests"],
    "all": ["history", "labs_and_tests", "medications"],
}
TRUNC = 1500
PERSONA = "You are an experienced emergency medicine physician."
COMMIT_INSTRUCTION = (
    'Give your final differential as ONLY a JSON object {"differential": ["dx1", ...]} '
    "with up to 10 concise diagnosis names, most likely first, and no other text."
)
COMMIT_FORCE = (
    'You must now give your final answer. Respond with ONLY {"differential": ["dx1", ...]} '
    "(up to 10, most likely first)."
)


def reveal_block(case: dict, item: str) -> str:
    content = case["revealable"].get(item, "(not available)")[:TRUNC]
    return f"{ITEM_TITLE[item]}:\n{content}"


def _parse_obj(text: str) -> dict | None:
    import re
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("action" in obj or "differential" in obj):
            return obj
    return None


def _clean(items) -> list[str]:
    return [str(x).strip() for x in (items or []) if str(x).strip()]


class HFChat:
    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

    def _render(self, convo: list[dict]) -> str:
        try:
            return self.tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Templates that forbid a separate system role (e.g. BioMistral): merge
            # any pending system text into the next user message.
            merged, sys_txt = [], ""
            for m in convo:
                if m["role"] == "system":
                    sys_txt = m["content"]
                elif m["role"] == "user":
                    merged.append({"role": "user",
                                   "content": (sys_txt + "\n\n" + m["content"]).strip()})
                    sys_txt = ""
                else:
                    merged.append(m)
            return self.tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def chat_batch(self, convos: list[list[dict]], max_new_tokens: int, batch_size: int) -> list[str]:
        out: list[str] = []
        for s in range(0, len(convos), batch_size):
            chunk = convos[s : s + batch_size]
            prompts = [self._render(c) for c in chunk]
            enc = self.tok(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out


# ----------------------------- oracle mode -----------------------------
def oracle_messages(case: dict, items: list[str]) -> list[dict]:
    parts = [case["s0"]] + [reveal_block(case, it) for it in items if it in case["revealable"]]
    user = "\n\n".join(parts) + "\n\n" + COMMIT_INSTRUCTION
    return [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}]


def run_oracle(chat: HFChat, cases: list[dict], batch_size: int, out: Path) -> None:
    jobs = [(c, cond) for c in cases for cond in CONDITIONS]
    convos = [oracle_messages(c, CONDITIONS[cond]) for c, cond in jobs]
    print(f"[oracle] generating {len(convos)} (cases={len(cases)} x conds={len(CONDITIONS)})")
    texts = chat.chat_batch(convos, 512, batch_size)
    with out.open("w") as f:
        for (c, cond), t in zip(jobs, texts):
            f.write(json.dumps({
                "stay_id": c["stay_id"], "condition": cond, "gold": c["gold"],
                "differential": parse_differential(t),
            }) + "\n")
    print(f"[oracle] wrote {len(jobs)} records -> {out}")


# ----------------------------- agent mode (batched) -----------------------------
def agent_system(case: dict) -> str:
    menu = ", ".join(k for k in ei.REVEALABLE_KEYS if k in case["revealable"])
    return (
        f"{PERSONA} You begin with limited information and may first REQUEST additional "
        "evidence, then give a final differential diagnosis.\n"
        f"Evidence you may request (exact keys): {menu}.\n"
        f"You may make at most {case['_budget']} requests. Request only evidence you expect "
        "to change your diagnosis, and stop as soon as you are confident.\n"
        "Each turn respond with ONLY one JSON object:\n"
        '  to request evidence: {"action": "request", "item": "<key>"}\n'
        '  to finish: {"differential": ["dx1", ...]}  (up to 10, most likely first)'
    )


def run_agent(chat: HFChat, cases: list[dict], budget: int, batch_size: int, out: Path) -> None:
    states = []
    for case in cases:
        case["_budget"] = budget
        states.append({
            "case": case,
            "convo": [
                {"role": "system", "content": agent_system(case)},
                {"role": "user", "content": case["s0"] + "\n\nWhat is your next action?"},
            ],
            "requests": [], "revealed": set(), "diff": [],
            "committed": False, "forced": False, "done": False,
        })

    max_turns = budget + 2
    for turn in range(max_turns):
        active = [i for i, s in enumerate(states) if not s["done"]]
        if not active:
            break
        texts = chat.chat_batch([states[i]["convo"] for i in active], 384, batch_size)
        for i, text in zip(active, texts):
            s = states[i]
            case = s["case"]
            s["convo"].append({"role": "assistant", "content": text})
            obj = _parse_obj(text)
            # commit?
            if obj and ("differential" in obj or obj.get("action") == "commit"):
                s["diff"] = _clean(obj.get("differential", []))
                s["committed"] = True
                s["done"] = True
                continue
            if s["forced"]:  # we already asked to commit; salvage whatever we got
                s["diff"] = parse_differential(text)
                s["done"] = True
                continue
            # request?
            if obj and obj.get("action") == "request" and len(s["requests"]) < budget:
                item = obj.get("item")
                if item in case["revealable"] and item not in s["revealed"]:
                    s["revealed"].add(item)
                    s["requests"].append(item)
                    s["convo"].append({"role": "user",
                                       "content": reveal_block(case, item) + "\n\nWhat is your next action?"})
                    continue
            # invalid / exhausted -> force a commit next turn
            s["convo"].append({"role": "user", "content": COMMIT_FORCE})
            s["forced"] = True
        print(f"[agent] turn {turn + 1}/{max_turns} | active={len(active)}")

    for s in states:  # finalize stragglers
        if not s["done"]:
            last = s["convo"][-1]
            s["diff"] = parse_differential(last["content"]) if last["role"] == "assistant" else []
            s["done"] = True

    with out.open("w") as f:
        for s in states:
            f.write(json.dumps({
                "stay_id": s["case"]["stay_id"], "gold": s["case"]["gold"],
                "requests": s["requests"], "n_requests": len(s["requests"]),
                "committed": s["committed"], "differential": s["diff"],
            }) + "\n")
    print(f"[agent] wrote {len(states)} records -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "agent"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = ei.load_interactive_cases(n=args.n, seed=args.seed)
    chat = HFChat(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "oracle":
        run_oracle(chat, cases, args.batch_size, out)
    else:
        run_agent(chat, cases, args.budget, args.batch_size, out)


if __name__ == "__main__":
    main()
