"""Single CLI entrypoint for PROVENA-MED.

Subcommands dispatch to ``main()`` of the underlying runner module, so anything
reachable via ``provena-med <verb> <noun>`` is also reachable as
``python -m provena_med.<group>.<module>`` with the same arguments.

  provena-med build      <cohort|rules|split|pack>   ...   # construct the dataset
  provena-med generate   <task>                      ...   # run a model
  provena-med eval       <axis>                      ...   # score outputs
  provena-med leaderboard                            ...   # aggregate panel results
  provena-med info                                         # version + layout

Each subcommand prints its own ``--help``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from provena_med import __version__

# verb -> noun -> dotted module path (module must expose main())
SUBCOMMANDS: dict[str, dict[str, str]] = {
    "build": {
        # cohorts
        "icu_mm":     "provena_med.data.build_icu_cohort",
        "eicu":       "provena_med.data.build_eicu_cohort",
        "mimic3":     "provena_med.data.build_mimic3_cohort",
        "ed_cardiac": "provena_med.data.build_provena_med_v2",
        # supporting
        "rules":      "provena_med.data.build_rules_fda",
        "split":      "provena_med.data.build_eval_split",
        "pack":       "provena_med.data.save_provena_med",
        "triangulate":"provena_med.data.triangulate",
        # enrichers (run after the per-cohort builder to add labs/meds/etc.)
        "labs_ts":    "provena_med.data.enrich_labs_ts",
        "meds":       "provena_med.data.enrich_meds",
        "demographics":"provena_med.data.enrich_demographics",
        "cxr_findings":"provena_med.data.run_cxr_findings",
    },
    "generate": {
        "staged":      "provena_med.tasks.run_generate_staged",
        "staged_vlm":  "provena_med.tasks.run_generate_staged_vlm",
        "diagnosis":   "provena_med.tasks.run_generate_dx",
        "safety":      "provena_med.tasks.run_generate_safety",
        "interactive": "provena_med.tasks.run_interactive",
        "oracle":      "provena_med.tasks.run_oracle_staged",
    },
    "eval": {
        "w":           "provena_med.eval.score_provenance",        # validity + sufficiency
        "panel":       "provena_med.eval.score_panel",             # 5-cohort W aggregate
        "diagnosis":   "provena_med.eval.score_dx_panel",          # Hit@1/3/5
        "safety":      "provena_med.eval.score_safety",            # unsafe-prescribing rate
        "interactive": "provena_med.eval.score_interactive",       # EES + Hit@k under budget
        "oracle":      "provena_med.eval.score_oracle_prov",       # oracle-reveal W
        "causal":      "provena_med.eval.causal_mmais",            # Shapley / necessity / sufficiency
        "m_probe":     "provena_med.eval.probe_m",                 # attention-knockout Delta^M
        "wxm":         "provena_med.eval.panel_judge",             # W x M panel (judge phase)
        "quadrant":    "provena_med.eval.quadrant_mxw",            # W x M quadrants
    },
}


def _print_info() -> None:
    print(f"provena-med {__version__}")
    print()
    print("verbs:")
    for verb, nouns in SUBCOMMANDS.items():
        print(f"  {verb}")
        for n in sorted(nouns):
            print(f"    {n:14s}  ({nouns[n]})")
    print("    leaderboard     (aggregate panel results into one row per model)")
    print()
    print("Each subcommand forwards remaining args to the underlying runner's argparse.")
    print("Get per-subcommand help with:  provena-med <verb> <noun> --help")


def _leaderboard(argv: list[str]) -> int:
    """Stub for the leaderboard aggregator. The released code consumes the JSONL outputs of
    the panel runs and emits a single CSV/JSON row per model. We dispatch to score_panel
    for the W aggregate, score_dx_panel for Hit@k, and provena_med.eval.panel_judge for
    the W x M cells; combining them into one leaderboard.json is straightforward and is
    the subject of ``scripts/aggregate_leaderboard.py`` (see docs/REPRODUCE.md)."""
    print("provena-med leaderboard: see docs/REPRODUCE.md for the panel aggregation script.")
    print("Quick recipe (after running the full panel via scripts/launch_*.sh):")
    print("  python -m provena_med.eval.score_panel --in outputs/ --out panel_w.jsonl")
    print("  python -m provena_med.eval.score_dx_panel --in 'outputs/dx_*.jsonl' \\")
    print("        --out panel_dx.jsonl")
    print("  python -m provena_med.eval.panel_judge --cohort cardiac_mm \\")
    print("        --ids gemma3_27b medgemma_27b ...  # populates W x M")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_info()
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    if argv[0] == "info":
        _print_info()
        return 0
    if argv[0] == "leaderboard":
        return _leaderboard(argv[1:])

    verb = argv.pop(0)
    if verb not in SUBCOMMANDS:
        print(f"unknown verb: {verb!r}; expected one of "
              f"{['info', 'leaderboard', *SUBCOMMANDS]}", file=sys.stderr)
        return 2
    nouns = SUBCOMMANDS[verb]
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: provena-med {verb} <noun> [options]\n\n{verb} nouns:")
        for n in sorted(nouns):
            print(f"  {n:14s}  ({nouns[n]})")
        print(f"\nGet per-noun help with:  provena-med {verb} <noun> --help")
        return 0
    noun = argv.pop(0)
    if noun not in nouns:
        print(f"unknown {verb} target: {noun!r}; expected one of {sorted(nouns)}",
              file=sys.stderr)
        return 2

    # Hand off to the underlying runner. Many runners use argparse on sys.argv directly,
    # so we splice argv[1:] in front of theirs.
    mod_path = nouns[noun]
    saved = sys.argv
    sys.argv = [mod_path, *argv]
    try:
        mod = importlib.import_module(mod_path)
        if not hasattr(mod, "main"):
            print(f"runner {mod_path} has no main() entrypoint", file=sys.stderr)
            return 2
        rc = mod.main()
        return rc if isinstance(rc, int) else 0
    finally:
        sys.argv = saved


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
