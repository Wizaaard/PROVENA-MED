"""Phase 0 of the true-pixel provenance path: turn real CXR pixels into IMAGE_FINDING units.

CheXagent reads each radiograph (DICOM -> pixels) and emits radiographic findings. We
split them into atomic finding sentences and cache them, keyed by hadm_id, as the
*pixel-derived* IMAGE_FINDING evidence units. The downstream reasoning LLM then cites
these (run_generate_staged.py --cohort cxr_pixel), so provenance is grounded in actual
pixels rather than in the radiologist's report text. CheXagent never sees the gold dx.

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_cxr_findings.py --n 50 \
      --out outputs/cxr_pixel_findings.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.core.bundle import split_sentences  # noqa: E402

CHEXAGENT = "~/models/CheXagent-8b"
# Anatomy-anchored prompts -> distinct, atomic, citable findings (CheXagent is tuned for these).
FINDING_PROMPTS = [
    "Describe the findings.",
    "Describe the cardiac silhouette and mediastinum.",
    "Describe the lungs and pleural spaces.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="cardiac_cxr", choices=["cardiac_cxr", "icu"],
                    help="which cohort's radiographs to read (both expose id + dicom_path)")
    ap.add_argument("--out", default="outputs/cxr_pixel_findings.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

    from provena_med.core.cxr_image import load_dicom_image
    if args.source == "icu":
        from provena_med.core.cohort_icu import load_icu_cases as load_cases_fn
    else:
        from provena_med.core.cohort_cxr import load_cxr_cases as load_cases_fn

    dev, dt = "cuda", torch.bfloat16
    proc = AutoProcessor.from_pretrained(CHEXAGENT, trust_remote_code=True)
    gen_cfg = GenerationConfig.from_pretrained(CHEXAGENT)
    model = AutoModelForCausalLM.from_pretrained(
        CHEXAGENT, torch_dtype=dt, trust_remote_code=True).to(dev).eval()

    def gen(img, prompt: str) -> str:
        inp = proc(images=[img], text=f" USER: <s>{prompt} ASSISTANT: <s>",
                   return_tensors="pt").to(dev, dt)
        out = model.generate(**inp, generation_config=gen_cfg,
                             max_new_tokens=args.max_new_tokens)[0]
        return proc.tokenizer.decode(out, skip_special_tokens=True).strip()

    cases = [c for c in load_cases_fn(args.n, args.seed) if c.get("dicom_path")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_units = 0
    with out.open("w") as f:
        for i, c in enumerate(cases):
            img = load_dicom_image(c["dicom_path"])
            seen, findings = set(), []
            for p in FINDING_PROMPTS:
                for s in split_sentences(gen(img, p)):
                    key = s.lower()
                    if key not in seen and len(s) >= 12:
                        seen.add(key)
                        findings.append(s)
            f.write(json.dumps({
                "id": c["id"], "subject_id": c["subject_id"],
                "dicom": c["dicom_path"].split("/files/")[-1],
                "image_findings": findings,
            }) + "\n")
            n_units += len(findings)
            if (i + 1) % 10 == 0:
                print(f"[cxr-findings] {i + 1}/{len(cases)}  ({n_units} units so far)")
    print(f"[cxr-findings] wrote {len(cases)} cases, {n_units} pixel IMAGE_FINDING units -> {out}")


if __name__ == "__main__":
    main()
