# Synthetic fixtures

These five fully-synthetic cases let you smoke-test the package end-to-end
**without** PhysioNet access. No real patient data — diagnoses and labs are
hand-crafted to be plausible but contain no PHI.

| file | purpose |
|---|---|
| `synthetic_cases.jsonl`  | five cases in the `provena-med` bundle format (header + typed evidence units + gold) |
| `synthetic_outputs.jsonl`| five matching model "staged artifact" outputs in the JSON schema the eval expects |

## Quick smoke test (no model required)

```bash
# verify install + package imports
python -c "import provena_med; print('provena-med', provena_med.__version__)"

# verify the CLI loads
provena-med info

# verify the bundle and claim parsers handle the fixtures
python - <<'PY'
import json
from provena_med.core.mmais import extract_claims_lenient
for line in open("examples/synthetic_outputs.jsonl"):
    rec = json.loads(line)
    claims = extract_claims_lenient(rec["output"])
    print(f"case {rec['stay_id']}: {len(claims)} claims parsed, "
          f"first claim cites {claims[0]['evidence'] if claims else []}")
PY
```

You should see `5` claims (or more) extracted with their cited evidence IDs.

## Score the fixtures (no model, no judge)

For the W axis, validity is judge-free (you only need the bundle):

```bash
python -m provena_med.eval.score_panel --inputs examples/synthetic_outputs.jsonl
```

For judge-supported precision (sufficiency), supply a HuggingFace judge id; this
will download the judge on first use. Use a small judge for the smoke test:

```bash
python -m provena_med.eval.score_provenance \
    --inputs examples/synthetic_outputs.jsonl \
    --judge meta-llama/Llama-3.2-3B-Instruct
```

(In paper experiments we use `meta-llama/Llama-3.3-70B-Instruct` as the held-out
judge; any larger instruct model works.)
