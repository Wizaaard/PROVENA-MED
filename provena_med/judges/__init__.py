"""Held-out LLM judge for set-support S(c, A). The default judge in the paper is
Llama-3.3-70B-Instruct, served via HuggingFace transformers. Closed-weight judges
work too: any model with ``.generate()`` and a fixed deterministic output schema."""

from provena_med.core.llm_judge import LLMJudge

__all__ = ["LLMJudge"]
