"""Shared diagnosis parsing + embedding-match scoring for PROVENA-MED."""
from __future__ import annotations

import json
import re

import numpy as np


def parse_differential(text: str) -> list[str]:
    """Extract a ranked diagnosis list from a model response (robust to extra text)."""
    if not text:
        return []
    for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "differential" in obj and isinstance(obj["differential"], list):
            return [str(x).strip() for x in obj["differential"] if str(x).strip()]
    m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return []


def bootstrap_ci(vals, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hit_recall(pairs: list[tuple[list[str], list[str]]], embedder, ks, tau: float) -> dict:
    """pairs: list of (predicted_list, gold_primary_list).

    Returns aggregate Hit@k / Recall@k plus per-case arrays (for bootstrapping)."""
    terms, idx = [], []
    for i, (preds, gold) in enumerate(pairs):
        for j, p in enumerate(preds):
            idx.append((i, "p", j))
            terms.append(p)
        for j, g in enumerate(gold):
            idx.append((i, "g", j))
            terms.append(g)
    n = len(pairs)
    if not terms:
        zero = {k: [0.0] * n for k in ks}
        return {"n": n, "hit": {k: 0.0 for k in ks}, "recall": {k: 0.0 for k in ks},
                "hit_pc": zero, "rec_pc": {k: [0.0] * n for k in ks}}
    emb = embedder.encode(terms, normalize_embeddings=True, show_progress_bar=False)
    vec: dict = {}
    for (i, kind, j), e in zip(idx, emb):
        vec.setdefault(i, {"p": {}, "g": {}})[kind][j] = e

    hit_pc = {k: [0.0] * n for k in ks}
    rec_pc = {k: [0.0] * n for k in ks}
    for i, (preds, gold) in enumerate(pairs):
        if not preds or not gold:
            continue
        pv, gv = vec[i]["p"], vec[i]["g"]
        for k in ks:
            top = [pv[t] for t in range(min(k, len(preds)))]
            if not top:
                continue
            top = np.stack(top)
            matched = sum(1 for j in range(len(gold)) if (top @ gv[j]).max() >= tau)
            hit_pc[k][i] = 1.0 if matched else 0.0
            rec_pc[k][i] = matched / len(gold)
    return {
        "n": n,
        "hit": {k: float(np.mean(hit_pc[k])) for k in ks},
        "recall": {k: float(np.mean(rec_pc[k])) for k in ks},
        "hit_pc": hit_pc,
        "rec_pc": rec_pc,
    }
