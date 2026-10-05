"""Part D: how predictions are scored against the ground truth.

Label harmonisation (lumbar/labels.py)
  TotalSegmentator vertebrae_L1..L5 and VerSe labels 20-24 both become ids 1-5.
  VerSe 25 (L6) and 28 (T13) get their own ids 8 and 9: they are never counted as
  L5 / T12 or as background. Our network has no L6 class, so in an L6 patient the
  scored levels are still L1-L5 and the L6 case is also reported on its own.

Scoring rules (decided before looking at results)
  - Dice per level on the ORIGINAL CT grid, the grid the annotation was made on.
  - A level is scored in a case only if the ground truth contains it and it is
    not a label scrap under 2 ml (FRAGMENT in the Part A audit).
  - Level in the prediction but not in the ground truth: a false-positive level;
    counted separately, not as a Dice of 0 (Dice is undefined without ground truth).
  - Cut-off vertebrae (TRUNCATED) are scored, and also reported without them.
  - Shifted Dice: the same Dice after renaming every predicted level one step up
    or down (T12 <- L1 <- L2 ... <- L5 <- sacrum). If a shift scores much higher than
    the strict Dice, the model found the vertebrae but counted them wrong.
  - Resampling ceiling: the ground truth sent to the model grid (2 or 1.5 mm) and
    back, compared with itself. No model working at that voxel size can beat it.
"""

from __future__ import annotations

import numpy as np

from .labels import NAME_TO_ID

# head-to-foot order of the classes the network knows; used for the shifted Dice
LEVEL_ORDER = [NAME_TO_ID[n] for n in ("T12", "L1", "L2", "L3", "L4", "L5", "sacrum")]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else float("nan")


def shifted_level(level_id: int, shift: int) -> int | None:
    """The class 'shift' steps further down the spine (+1 = towards the sacrum)."""
    i = LEVEL_ORDER.index(level_id) + shift
    return LEVEL_ORDER[i] if 0 <= i < len(LEVEL_ORDER) else None
