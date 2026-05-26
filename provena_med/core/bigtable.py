"""Shared helper for streaming huge MIMIC/eICU event tables with an awk pre-filter.

The clinical event tables (chartevents ~430M rows, eICU lab/vitalPeriodic ~100M+) are far
too large to read whole into pandas. We pre-filter with a single streaming awk pass that
keeps only rows whose id column is in a target set (and optionally whose item column
matches a small itemid/name set), then hand the much smaller result to pandas.

NOTE: awk is line-based, so this is only safe for tables whose key columns precede any
free-text field that can contain commas/newlines (true for chartevents, labevents, eICU
lab/vitals). For tables with embedded multiline text (MIMIC-III NOTEEVENTS) use pandas
chunked reads instead.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def awk_filter(table: Path, idfile: Path, id_field: int, out: Path,
               item_field: int | None = None, items=None, regex: bool = True) -> pd.DataFrame:
    """Keep rows of gz `table` where field `id_field` is listed in `idfile`
    (one id per line) and, if given, field `item_field` matches `items`.
    `items` matched as exact-anchored regex (numeric itemids) or as a literal set."""
    conds = [f"(${id_field} in s)"]
    awk_args = ["awk", "-F,"]
    if item_field and items:
        if regex:
            items_re = "^(" + "|".join(str(i) for i in items) + ")$"
            awk_args += ["-v", f"items={items_re}"]
            conds.append(f"(${item_field} ~ items)")
        else:
            # literal membership against a second set loaded from items (file of names)
            raise NotImplementedError("use regex=True")
    prog = (f'NR==FNR{{s[$1]=1; next}} FNR==1{{print; next}} ' + " && ".join(conds))
    awk_args += [prog, str(idfile), "-"]
    with out.open("wb") as fout:
        p1 = subprocess.Popen(["zcat", str(table)], stdout=subprocess.PIPE)
        p2 = subprocess.Popen(awk_args, stdin=p1.stdout, stdout=fout)
        p1.stdout.close()
        p2.communicate()
    return pd.read_csv(out)


def write_ids(ids, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(sorted(set(int(x) for x in ids))).to_csv(path, index=False, header=False)
