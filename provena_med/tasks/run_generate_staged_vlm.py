"""TRUE multimodal staged generation: a vision-language model SEES the chest radiograph.

Runs the same staged provenance task as run_generate_staged, but for VLM-capable models
(gemma-3, medgemma) and -- in pixel mode -- passes the actual DICOM image alongside the
text bundle. Two conditions on the same cases isolate the value of perception:
  --image off : text-only (IMAGE_FINDING units are the radiologist report text)  [condition T]
  --image on  : the model also sees the radiograph pixels                         [condition P]
The bundle (and thus the citable IDs / MM-AIS scoring) is identical across conditions.

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python run_generate_staged_vlm.py \
      --model .../gemma-3-12b-it --cohort icu_mm --n 50 --image --out outputs/vlm_icu_P_gemma3_12b.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from provena_med import DATA_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provena_med.tasks.run_generate_staged import PERSONA, build_messages, load_provena  # noqa: E402
from provena_med.core.cxr_image import load_dicom_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", default="icu_mm")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", action="store_true", help="pixel condition: also show the radiograph")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cases = load_provena(args.cohort, args.n, args.seed)
    # need dicom_path per case for the pixel condition -> map from the cohort cache
    dicom = {}
    if args.image:
        src = DATA_ROOT / "PROVENA-MED/v0.2/cohorts" / f"{args.cohort}.jsonl"
        for line in open(src):
            r = json.loads(line)
            dicom[int(r["id"])] = r.get("dicom_path", "")

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def gen(user_text: str, img) -> str:
        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": PERSONA + "\n\n" + user_text})
        msgs = [{"role": "user", "content": content}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        return proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    outs = []
    for i, c in enumerate(cases):
        # reuse the exact text prompt the text panel used (system folded into user content)
        user_text = build_messages(c["header"], c["bundle"])[1]["content"]
        img = None
        if args.image:
            dp = dicom.get(int(c["id"]), "")
            img = load_dicom_image(dp) if dp else None
        outs.append(gen(user_text, img))
        if (i + 1) % 10 == 0:
            print(f"[vlm:{'P' if args.image else 'T'}:{args.cohort}] {i + 1}/{len(cases)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c, text in zip(cases, outs):
            f.write(json.dumps({"stay_id": c["id"], "gold": c["gold"],
                                "bundle": c["bundle"], "output": text}) + "\n")
    n_img = sum(1 for c in cases if args.image and dicom.get(int(c["id"])))
    print(f"[vlm:{'P' if args.image else 'T'}] wrote {len(cases)} (images shown: {n_img}) -> {out}")


if __name__ == "__main__":
    main()
