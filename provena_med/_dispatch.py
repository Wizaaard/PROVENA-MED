"""Dispatch helper for cluster scripts:
    python -m provena_med._dispatch <name> [args...]

forwards to whichever runner module exposes that ``name`` (either a CLI noun
from :mod:`provena_med.cli` or a raw runner module name like
``run_generate_staged``). The SBATCH templates in ``scripts/`` use this so they
do not have to spell out the full subpackage path of every runner.

End-users running interactively should prefer the ``provena-med`` CLI directly.
"""
from __future__ import annotations

import importlib
import sys

from provena_med.cli import SUBCOMMANDS

# Flat noun -> module map across all verbs (used by both the CLI and the dispatcher).
_FLAT = {n: m for verb in SUBCOMMANDS for n, m in SUBCOMMANDS[verb].items()}

# Allowed bare prefixes for raw module names (so we can locate them by trying each
# of the three runner subpackages).
_SUBPKGS = ("data", "tasks", "eval")
_BARE_PREFIXES = ("build_", "enrich_", "run_", "save_", "score_",
                  "causal_", "panel_", "probe_", "quadrant_", "triangulate")


def _resolve(name: str) -> str | None:
    if name in _FLAT:
        return _FLAT[name]
    if name.startswith(_BARE_PREFIXES):
        for subp in _SUBPKGS:
            cand = f"provena_med.{subp}.{name}"
            try:
                importlib.import_module(cand)
                return cand
            except ImportError:
                continue
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m provena_med._dispatch <name> [args...]", file=sys.stderr)
        return 2
    name = sys.argv[1]
    target = _resolve(name)
    if target is None:
        print(f"unknown dispatch target: {name!r}", file=sys.stderr)
        return 2
    # rewrite argv so the target's argparse sees the right program name + remaining args
    sys.argv = [target, *sys.argv[2:]]
    mod = importlib.import_module(target)
    if not hasattr(mod, "main"):
        print(f"target {target} has no main() entrypoint", file=sys.stderr)
        return 2
    rc = mod.main()
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
