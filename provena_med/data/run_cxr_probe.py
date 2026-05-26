"""Prove the pixel path: real MIMIC-CXR DICOM -> image -> CheXagent (a CXR VLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from provena_med.core.cohort_cxr import load_cxr_cases
from provena_med.core.cxr_image import load_dicom_image

M = "~/models/CheXagent-8b"
dev, dt = "cuda", torch.bfloat16

proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
gen_cfg = GenerationConfig.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=dt, trust_remote_code=True).to(dev).eval()

cases = load_cxr_cases(2, seed=0)
for c in cases:
    img = load_dicom_image(c["dicom_path"])
    for prompt in ["Describe the findings.",
                   "List the most likely cardiopulmonary diagnoses based on this chest X-ray."]:
        inputs = proc(images=[img], text=f" USER: <s>{prompt} ASSISTANT: <s>",
                      return_tensors="pt").to(device=dev, dtype=dt)
        out = model.generate(**inputs, generation_config=gen_cfg)[0]
        resp = proc.tokenizer.decode(out, skip_special_tokens=True)
        print(f"\n[hadm {c['id']}] PROMPT: {prompt}\n  -> {resp[:300]}")
    print(f"  GOLD dx: {c['gold'][:2]}")
print("\nPIXEL PATH OK")
