"""PROVENA-MED: a clinician-free, provenance-faithful, multimodal clinical-reasoning benchmark."""

import os
from pathlib import Path

__version__ = "0.1.0"

# Dataset locations are intentionally configured outside the package because the
# credentialed PhysioNet inputs cannot be redistributed with this repository.
DATA_ROOT = Path(os.environ.get("PROVENA_DATA_ROOT", ".")).expanduser()


def require_data_root() -> Path:
    """Return the configured data root or raise an actionable configuration error."""
    if not os.environ.get("PROVENA_DATA_ROOT"):
        raise RuntimeError(
            "PROVENA_DATA_ROOT is required for build and generation commands. "
            "Set it to the directory containing Datasets/ and PROVENA-MED/."
        )
    return DATA_ROOT
