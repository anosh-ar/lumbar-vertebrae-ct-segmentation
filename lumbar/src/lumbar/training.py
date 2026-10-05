"""Part B: training data, patch sampling and loss.

Which cases we train on (choose_training_cases)
  All TotalSegmentator v3 *train* crops from step 2, except
  - NO_LUMBAR: no L1-L5 at all (thorax-only or pelvis-only crops),
  - FRAGMENT: a lumbar level is only a scrap under 2 ml (thin slivers at the
    edge of the scan; the label is not a usable example of that vertebra).
  Cut-off vertebrae (TRUNCATED) and metal are kept: real scans have them too.
  The official test split is never touched. Because v3 has no validation split,
  10 % of the chosen cases (fixed random seed) are held out to pick the best
  checkpoint.

Patches (sample_patch)
  The crops are small (median 86 x 82 x 152 voxels at 2 mm, about 2.4x more voxels
  at 1.5 mm: 3.5 GB vs 8.2 GB for all cases), so all of them fit in RAM.
  A patch is taken at a random position; with probability 2/3 it is centred on
  a random voxel of a randomly chosen lumbar level that is present, so rare
  levels at the crop edges (L1, L5) are seen as often as the middle ones.
  No other augmentation (no flips, rotations or intensity changes): with ~1000
  training cases and a short run they are not needed, and every one would be
  another choice to justify. Flips would also break the level order (head-foot)
  or create mirrored anatomy (left-right).

Loss (dice_plus_cross_entropy)
  Cross-entropy (per voxel) + soft Dice (per class, over the whole batch), the
  standard nnU-Net pair: cross-entropy gives smooth gradients, Dice keeps small
  classes from being ignored next to the large background.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .labels import NUM_TRAIN_CLASSES, TARGET_IDS
from .sources import QC_REPORTS_DIR, preprocessed_folder

EXCLUDED_FLAGS = ("NO_LUMBAR", "FRAGMENT")
VALIDATION_FRACTION = 0.10
FOREGROUND_PATCH_PROBABILITY = 2 / 3


def choose_training_cases(source: str = "totalseg_v3_train", spacing: float = 2.0, seed: int = 0,
                          same_cases_as=None):
    """(training case ids, validation case ids, table of every case with the reason
    it was kept or excluded).

    same_cases_as: a training_cases.csv of an earlier run. Its train/validation lists
    are reused exactly (for a fair comparison between runs), after checking that every
    case also exists in this voxel size's preprocessed folder."""
    folder = preprocessed_folder(source, spacing, "lumbar")
    if same_cases_as is not None:
        t = pd.read_csv(same_cases_as)
        done = pd.read_csv(folder / "preprocessing_status.csv")
        written = set(done.loc[done["status"].isin(["written", "already_done"]), "case_id"])
        used = t.loc[t["use"].isin(["train", "validation"]), "case_id"]
        missing = sorted(set(used) - written)
        if missing:
            raise ValueError(f"{len(missing)} cases of {same_cases_as} are not preprocessed in {folder}: {missing[:5]}")
        return (sorted(t.loc[t["use"] == "train", "case_id"]), sorted(t.loc[t["use"] == "validation", "case_id"]), t)
    status = pd.read_csv(folder / "preprocessing_status.csv")
    flags = pd.read_csv(QC_REPORTS_DIR / f"anomalies_per_case_{source}.csv")[["case_id", "flags"]]
    t = status.merge(flags, on="case_id", how="left")
    t["flags"] = t["flags"].fillna("")
    t["use"] = "train"
    t.loc[~t["status"].isin(["written", "already_done"]), "use"] = "excluded: " + t["status"]
    for f in EXCLUDED_FLAGS:
        hit = (t["use"] == "train") & t["flags"].str.split().map(lambda fl, f=f: f in fl)
        t.loc[hit, "use"] = f"excluded: {f}"
    chosen = t.loc[t["use"] == "train", "case_id"].sort_values().to_numpy()
    rng = np.random.default_rng(seed)
    val = set(rng.choice(chosen, size=int(round(len(chosen) * VALIDATION_FRACTION)), replace=False))
    t.loc[t["case_id"].isin(val), "use"] = "validation"
    return sorted(set(chosen) - val), sorted(val), t


def load_cases_into_memory(case_ids, source: str = "totalseg_v3_train", spacing: float = 2.0) -> dict:
    """{case_id: (image float16, label uint8)} for the given preprocessed cases."""
    folder = preprocessed_folder(source, spacing, "lumbar")
    out = {}
    for cid in case_ids:
        d = np.load(folder / f"{cid}_image_and_label.npz")
        out[cid] = (d["image"], d["label"])
    return out


def pad_to_at_least(arr: np.ndarray, shape, value) -> np.ndarray:
    pad = [(0, max(0, s - a)) for a, s in zip(arr.shape, shape)]
    return np.pad(arr, pad, constant_values=value) if any(p[1] for p in pad) else arr


def sample_patch(image: np.ndarray, label: np.ndarray, patch_size, rng: np.random.Generator):
    """One (image, label) patch; see the module docstring for the rule."""
    image = pad_to_at_least(image, patch_size, 0)      # 0 = air after normalisation
    label = pad_to_at_least(label, patch_size, 0)
    present = [c for c in TARGET_IDS if (label == c).any()] if rng.random() < FOREGROUND_PATCH_PROBABILITY else []
    if present:
        vox = np.argwhere(label == rng.choice(present))
        centre = vox[rng.integers(len(vox))]
        start = [int(np.clip(c - p // 2, 0, s - p)) for c, p, s in zip(centre, patch_size, label.shape)]
    else:
        start = [int(rng.integers(0, s - p + 1)) for p, s in zip(patch_size, label.shape)]
    sl = tuple(slice(a, a + p) for a, p in zip(start, patch_size))
    return image[sl].astype(np.float32), label[sl].astype(np.int64)


def dice_plus_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + (1 - mean soft Dice over the foreground classes)."""
    ce = F.cross_entropy(logits, target)
    probs = torch.softmax(logits.float(), dim=1)
    onehot = F.one_hot(target, NUM_TRAIN_CLASSES).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)                                 # sum over batch and space
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + 1e-5) / (denom + 1e-5)
    return ce + (1 - dice[1:].mean())


def dice_per_level(pred: np.ndarray, gt: np.ndarray, ids=TARGET_IDS) -> dict[int, float]:
    """Hard Dice for every level present in the ground truth."""
    out = {}
    for c in ids:
        g = gt == c
        if g.any():
            p = pred == c
            out[c] = 2 * float((p & g).sum()) / float(p.sum() + g.sum())
    return out
