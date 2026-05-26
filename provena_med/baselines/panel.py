"""The 10-model open-weight baseline panel reported in the paper.

Each entry maps a stable ID (used in result filenames) to a HuggingFace model name.
Override locally with ``--model <hf_id_or_local_path>`` on any task CLI; the panel
mapping is only used by the SLURM panel launchers in ``scripts/``.

Closed-weight baselines (GPT-4V class) are evaluable on the W axis only and live
behind their respective inference APIs; they are not in this default panel.
"""
from __future__ import annotations

# Model ID -> (HuggingFace repo, family, clinical-specialized?)
PANEL: dict[str, tuple[str, str, bool]] = {
    # general
    "llama31_8b":    ("meta-llama/Llama-3.1-8B-Instruct",   "llama-3",   False),
    "llama32_3b":    ("meta-llama/Llama-3.2-3B-Instruct",   "llama-3",   False),
    "mistral7b":     ("mistralai/Mistral-7B-Instruct-v0.3", "mistral",   False),
    "gemma3_4b":     ("google/gemma-3-4b-it",               "gemma-3",   False),
    "gemma3_12b":    ("google/gemma-3-12b-it",              "gemma-3",   False),
    "gemma3_27b":    ("google/gemma-3-27b-it",              "gemma-3",   False),
    # clinical-specialized (the dagger row in the leaderboard)
    "med42_8b":      ("m42-health/Llama3-Med42-8B",         "llama-3",   True),
    "biomistral_7b": ("BioMistral/BioMistral-7B-DARE",      "mistral",   True),
    "medgemma_4b":   ("google/medgemma-4b-it",              "gemma-3",   True),
    "medgemma_27b":  ("google/medgemma-27b-it",             "gemma-3",   True),
}

# The held-out judge used in all W and W x M scoring. Held out by construction
# (the judge does not appear in the evaluated panel).
DEFAULT_JUDGE = "meta-llama/Llama-3.3-70B-Instruct"


def hf_id(model_id: str) -> str:
    """Resolve a panel ID (e.g. 'gemma3_27b') to its HuggingFace repo id."""
    if model_id in PANEL:
        return PANEL[model_id][0]
    # already a HF id or a local path
    return model_id


def is_clinical(model_id: str) -> bool:
    return PANEL.get(model_id, (None, None, False))[2]


def family(model_id: str) -> str | None:
    return PANEL.get(model_id, (None, None, False))[1]
