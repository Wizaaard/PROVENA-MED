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
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from provena_med import __version__, require_data_root

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
    """Aggregate task-specific scorer JSONL files into one row per model."""
    parser = argparse.ArgumentParser(prog="provena-med leaderboard")
    parser.add_argument("--in", dest="input_dir", default="outputs",
                        help="directory containing scorer JSONL outputs")
    parser.add_argument("--out", default="leaderboard.json",
                        help="JSON file to write")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    patterns = ("panel_scores*.jsonl", "dx_panel*.jsonl", "causalW_panel*.jsonl",
                "quadrants*.jsonl")
    files = sorted({path for pattern in patterns for path in input_dir.glob(pattern)})
    if not files:
        print(f"no scorer JSONL files found in {input_dir}", file=sys.stderr)
        return 2

    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add_numeric(model: str, prefix: str, value) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            metrics[model][prefix].append(float(value))
        elif isinstance(value, dict):
            for key, nested in value.items():
                add_numeric(model, f"{prefix}.{key}", nested)

    for path in files:
        prefix = path.stem
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            model = record.get("model")
            if not isinstance(model, str) or not model:
                print(f"skipping {path}:{line_no}: no model field", file=sys.stderr)
                continue
            for key, value in record.items():
                if key not in {"model", "cohort", "n", "parsed", "n_salient"}:
                    add_numeric(model, f"{prefix}.{key}", value)

    rows = [{"model": model,
             "metrics": {key: fmean(values) for key, values in sorted(values_by_key.items())}}
            for model, values_by_key in sorted(metrics.items())]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} model rows from {len(files)} scorer files -> {output}")
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

    if verb in {"build", "generate"} and not any(flag in argv for flag in ("-h", "--help")):
        try:
            require_data_root()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
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
