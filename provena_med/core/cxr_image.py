"""Load a MIMIC-CXR DICOM into an RGB PIL image for vision-language models."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from provena_med import DATA_ROOT

CXR_ROOT = str(DATA_ROOT / "Datasets/MIMIC-IV/mimic-cxr/2.1.0")


@lru_cache(maxsize=4096)
def load_dicom_image(path: str, size: int = 896) -> Image.Image:
    try:
        import pydicom
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "DICOM image support requires the vision dependencies: pip install 'provena-med[vision]'"
        ) from exc
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    a = (a - a.min()) / (np.ptp(a) + 1e-6)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = 1.0 - a  # invert so bones are bright
    img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    img.thumbnail((size, size))
    return img


def dicom_full_path(rel_path: str) -> str:
    """cxr-record-list 'path' is relative to the dataset root."""
    return f"{CXR_ROOT}/{rel_path}"


if __name__ == "__main__":
    import glob
    d = glob.glob(f"{CXR_ROOT}/files/p10/p*/s*/*.dcm")[0]
    im = load_dicom_image(d)
    print("loaded", d.split("/files/")[-1], "->", im.size, im.mode)
