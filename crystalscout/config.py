"""Central configuration: paths and scientific constants in one place."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CRYSTALSCOUT_DATA", ROOT / "data"))
OBELIX_DIR = DATA_DIR / "obelix"
MP_DIR = DATA_DIR / "materials_project"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
RUNS_DIR = DATA_DIR / "runs"

# Superionic threshold used by Sendek et al. 2019: RT conductivity >= 1e-4 S/cm
SUPERIONIC_THRESHOLD = 1e-4

RANDOM_SEED = 42

for _d in (PROCESSED_DIR, MODELS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
