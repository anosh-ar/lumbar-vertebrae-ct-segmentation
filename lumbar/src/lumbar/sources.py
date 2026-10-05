"""Named data sources and output locations.

Scripts say ``--source verse19_validation`` instead of repeating paths. With no
``--source``, every source whose files are on disk is processed. The data root
defaults to <project>/data and can be overridden with the LUMBAR_DATA
environment variable (needed on the GPU workstation).
"""

from __future__ import annotations

import os
from pathlib import Path

from .cases import Case, find_totalsegmentator_cases, find_verse_cases

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("LUMBAR_DATA", PROJECT_ROOT / "data"))
ARCHIVES_DIR = DATA_ROOT / "archives"
EXTRACTED_DIR = DATA_ROOT / "raw"
PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
QC_REPORTS_DIR = PROJECT_ROOT / "outputs" / "qc"
# trained network (step 4); lives inside lumbar/ so it ships with the code zip.
# WEIGHTS_FILE is the model the deliverable uses (trained at 2 mm); models trained at
# another voxel size get the size in their name (weights_file_for_spacing).
WEIGHTS_DIR = PROJECT_ROOT / "lumbar" / "weights"
WEIGHTS_FILE = WEIGHTS_DIR / "lumbar_vertebrae_resenc_unet.pt"


def weights_file_for_spacing(spacing: float) -> Path:
    """2 mm -> the existing lumbar_vertebrae_resenc_unet.pt; e.g. 1.5 mm -> ..._1.5mm.pt."""
    return WEIGHTS_FILE if spacing == 2.0 else WEIGHTS_DIR / f"lumbar_vertebrae_resenc_unet_{spacing:g}mm.pt"

TOTALSEG_V3_ZIP = ARCHIVES_DIR / "Totalsegmentator_dataset_v300.zip"

# name -> (location, how to find its cases). TotalSegmentator v3 is read
# straight from its zip, split by the official meta.csv split; VerSe from
# data/raw after 00_extract_verse_archives.py.
SOURCES = {
    "totalseg_v3_train":  (TOTALSEG_V3_ZIP, lambda r: find_totalsegmentator_cases(r, {"train"})),
    "totalseg_v3_test":   (TOTALSEG_V3_ZIP, lambda r: find_totalsegmentator_cases(r, {"test"})),
    "verse19_training":   (EXTRACTED_DIR / "verse19" / "training", lambda r: find_verse_cases(r, "verse19training")),
    "verse19_validation": (EXTRACTED_DIR / "verse19" / "validation", lambda r: find_verse_cases(r, "verse19validation")),
    "verse19_test":       (EXTRACTED_DIR / "verse19" / "test", lambda r: find_verse_cases(r, "verse19test")),
}


def load_cases(source: str) -> list[Case]:
    root, find_cases = SOURCES[source]
    if not root.exists():
        raise FileNotFoundError(f"source '{source}' not available: {root} does not exist")
    return find_cases(root)


def downloaded_sources() -> list[str]:
    """Every source whose files are on disk: the default for all scripts."""
    return [n for n, (root, _) in SOURCES.items() if root.exists()]


def preprocessed_folder(source: str, spacing: float, crop: str) -> Path:
    """Folder name records the settings, so caches with another voxel size or crop
    can never be mixed up: e.g. totalseg_v3_train_2mm_lumbar_crop, ..._1.5mm_lumbar_crop."""
    crop_name = {"lumbar": "lumbar_crop", "none": "full_image"}[crop]
    return PREPROCESSED_DIR / f"{source}_{spacing:g}mm_{crop_name}"
