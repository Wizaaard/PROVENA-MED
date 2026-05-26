"""PROVENA-MED provenance scoring (phase 2): MM-AIS over the staged artifacts."""
from __future__ import annotations

import argparse
import json

from provena_med.core.mmais import score_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.inp)]
    r = score_records(records, tau=args.tau)

    print(f"\n=== MM-AIS ({args.inp}) | n={r['n']} | tau={r['tau']} ===")
    print(f"  staged artifacts parsed     : {r['parsed']}/{r['n']}")
    print(f"  total claims                : {r['claims']}")
    print(f"  claims with valid citation  : {r['claims_with_valid_citation']}/{r['claims']}")
    print(f"  citation validity (ID exists): {r['citation_validity']:.3f}")
    print(f"  mean citations / claim      : {r['mean_citations_per_claim']:.2f}")
    print(f"  MM-AIS precision (overall)  : {r['mm_ais_precision']:.3f}  (n={r['n_scored_citations']} citations)")
    print(f"  MM-AIS precision by modality:")
    for m, v in sorted(r["mm_ais_precision_by_modality"].items()):
        print(f"      {m:12s} {v:.3f}")


if __name__ == "__main__":
    main()
