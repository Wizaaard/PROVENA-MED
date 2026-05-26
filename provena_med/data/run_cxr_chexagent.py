"""CheXagent on the cardiac x CXR image cohort: image-based findings + diagnosis.

Two-phase in one script: (1) CheXagent generates findings + a diagnosis list from the
real radiograph; (2) score the diagnosis list against ICD gold by BioLORD Hit@k.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/cxr_chexagent.jsonl")
    ap.add_argument("--use-hpi", action="store_true", help="also give the HPI text (image+text)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from provena_med.core.cohort_cxr import load_cxr_cases
    from provena_med.core.cxr_image import load_dicom_image

    M = "~/models/CheXagent-8b"
    dev, dt = "cuda", torch.bfloat16
    proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
    gen_cfg = GenerationConfig.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=dt, trust_remote_code=True).to(dev).eval()

    def gen(img, prompt, max_new=96):
        inp = proc(images=[img], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(dev, dt)
        out = model.generate(**inp, generation_config=gen_cfg, max_new_tokens=max_new)[0]
        return proc.tokenizer.decode(out, skip_special_tokens=True).strip()

    cases = load_cxr_cases(args.n, args.seed)
    recs = []
    for i, c in enumerate(cases):
        img = load_dicom_image(c["dicom_path"])
        dx_prompt = "List up to 5 likely diagnoses for this patient, most likely first, separated by semicolons."
        if args.use_hpi:
            dx_prompt = f"History: {str(c['HPI'])[:400]} {dx_prompt}"
        recs.append({"id": c["id"], "gold": c["gold"],
                     "findings": gen(img, "Describe the findings."),
                     "dx": gen(img, dx_prompt)})
        if (i + 1) % 10 == 0:
            print(f"[chexagent] {i + 1}/{len(cases)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    del model
    torch.cuda.empty_cache()

    # ---- score diagnosis (BioLORD Hit@k) ----
    from sentence_transformers import SentenceTransformer
    from provena_med.core.dxmatch import hit_recall
    emb = SentenceTransformer("FremyCompany/BioLORD-2023")
    pairs = [([s.strip() for s in re.split(r"[;\n]", r["dx"]) if s.strip()], r["gold"]) for r in recs]
    m = hit_recall(pairs, emb, [1, 3, 5], 0.75)
    print(f"\n=== CheXagent image-{'+HPI' if args.use_hpi else 'only'} diagnosis (cardiac x CXR, n={len(recs)}) ===")
    for k in [1, 3, 5]:
        print(f"  Hit@{k}={m['hit'][k]:.3f}  Recall@{k}={m['recall'][k]:.3f}")
    print("\nsample findings:")
    for r in recs[:3]:
        print(f"  [hadm {r['id']}] {r['findings'][:140]}")
        print(f"      dx: {r['dx'][:120]} | gold: {r['gold'][:2]}")


if __name__ == "__main__":
    main()
