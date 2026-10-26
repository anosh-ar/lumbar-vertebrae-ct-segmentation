"""NAS extension: the search space of the hardware-aware architecture search.

One "architecture" is a small dict of four choices:
  spacing_mm   voxel size the CT is resampled to (the "sensor" parameter: coarser
               voxels = fewer voxels = faster, but thin structures get lost)
  n_stages     resolution stages of the U-Net (depth)
  base_width   feature channels of the first stage; they double per stage, capped at 320 (width)
  blocks       residual blocks per encoder stage: "light" 1-1-2-2-2-2 or "M" 1-3-4-6-6-6
               (the nnU-Net ResEnc-M preset), first n_stages entries

Scaling depth, width and input resolution together follows EfficientNet
(Tan & Le, ICML 2019). The current deliverable model is one point of the space
(BASELINE_ARCHITECTURE), so it is measured with the same protocol as the others.

network_settings_for() turns an architecture into the keyword arguments of
build_lumbar_segmentation_network (lumbar/network.py); checkpoints store these
settings, so load_trained_network() rebuilds any searched model.
"""

from __future__ import annotations

import itertools
import json

from .labels import NUM_TRAIN_CLASSES
from .network import patch_size_for_spacing

# The task note lists 1.5, 2 and 3 mm. The overnight run searched only 1.5 and 3 mm
# (no time to also preprocess 2 mm); scripts take --spacings to choose.
ALL_SPACINGS_MM = (1.5, 2.0, 3.0)
SEARCHED_SPACINGS_MM = (1.5, 3.0)
N_STAGES = (4, 5, 6)
BASE_WIDTHS = (8, 16, 32)
ENCODER_BLOCKS = {"light": [1, 1, 2, 2, 2, 2], "M": [1, 3, 4, 6, 6, 6]}
MAX_FEATURES = 320

BASELINE_ARCHITECTURE = {"spacing_mm": 1.5, "n_stages": 6, "base_width": 32, "blocks": "M"}


def network_settings_for(arch: dict) -> dict:
    """Keyword arguments for build_lumbar_segmentation_network. The baseline gives
    exactly NETWORK_SETTINGS of the deliverable (32-64-128-256-320-320, 1-3-4-6-6-6)."""
    n = arch["n_stages"]
    return {
        "input_channels": 1,
        "n_stages": n,
        "features_per_stage": [min(arch["base_width"] * 2 ** i, MAX_FEATURES) for i in range(n)],
        "kernel_sizes": 3,
        "strides": [1] + [2] * (n - 1),
        "n_blocks_per_stage": ENCODER_BLOCKS[arch["blocks"]][:n],
        "n_conv_per_stage_decoder": [1] * (n - 1),
        "num_classes": NUM_TRAIN_CLASSES,
    }


def patch_size_for(arch: dict) -> tuple[int, int, int]:
    """Same patch in cm for every architecture (lumbar/network.py), so only the voxel
    size changes it: 1.5 mm -> 128x128x224, 2 mm -> 96x96x160, 3 mm -> 64x64x96
    (320 mm / 3 = 106.7 rounds down to 96 voxels = 28.8 cm). Multiples of 32 also
    fit networks with fewer stages (they halve the image only 3 or 4 times)."""
    return patch_size_for_spacing(arch["spacing_mm"])


def architecture_name(arch: dict) -> str:
    """Short readable name, e.g. 1.5mm_6stages_w32_M; used for files and plot labels."""
    return f"{arch['spacing_mm']:g}mm_{arch['n_stages']}stages_w{arch['base_width']}_{arch['blocks']}"


def all_architectures(spacings=SEARCHED_SPACINGS_MM) -> list[dict]:
    """Every architecture of the space for the given voxel sizes (18 per voxel size)."""
    return [{"spacing_mm": float(s), "n_stages": n, "base_width": w, "blocks": b}
            for s, n, w, b in itertools.product(spacings, N_STAGES, BASE_WIDTHS, ENCODER_BLOCKS)]


def read_architecture(text_or_path: str) -> dict:
    """An architecture from a JSON string or a JSON file (script 04 --architecture)."""
    text = text_or_path.strip()
    if not text.startswith("{"):
        with open(text, encoding="utf-8") as f:
            text = f.read()
    arch = json.loads(text)
    arch["spacing_mm"] = float(arch["spacing_mm"])
    return arch
