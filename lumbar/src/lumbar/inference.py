"""Part B: run the trained network on a CT of any size and orientation.

Steps for one CT
  1. bring the CT into the canonical space exactly as in training
     (preprocess.py: HU, smoothing where downsampled, RAS grid with the voxel size
     stored in the checkpoint (2 or 1.5 mm), HU window -> 0..1);
     the whole scan is used, because at inference there is no label to crop with
  2. slide a patch-sized window over the volume with 50 % overlap; average the
     class probabilities of overlapping windows, weighting each window's centre
     more than its border (border predictions see less context)
  3. take the most likely class per voxel
  4. clean-up: for every class keep only its largest connected piece. A vertebra
     is one connected bone, so smaller pieces with the same label are errors
     (e.g. a thoracic vertebra outside the training crop called "L1")
  5. resample the labels back onto the original CT grid (nearest neighbour)
"""

from __future__ import annotations

import itertools

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from .geometry import antialias_sigma_per_axis, ras_grid_covering_volume, resample_onto_grid
from .preprocess import AIR_HU, TARGET_SPACING, load_ct_as_hu, normalise_hu_to_0_1


def ct_to_model_grid(img: nib.Nifti1Image, spacing: float = TARGET_SPACING) -> tuple[np.ndarray, np.ndarray]:
    """Whole CT in the canonical space: (image 0..1 float32, affine voxel -> RAS mm).
    Same functions and order as preprocess_case_to_canonical, without the crop."""
    hu, _ = load_ct_as_hu(img)
    affine = img.affine
    sigma = antialias_sigma_per_axis(affine, spacing)
    if (sigma > 0).any():
        hu = ndimage.gaussian_filter(hu, sigma, mode="nearest")
    dst_affine, dst_shape = ras_grid_covering_volume(affine, hu.shape, spacing)
    hu_c = resample_onto_grid(hu, affine, dst_affine, dst_shape, order=1, cval=AIR_HU)
    return normalise_hu_to_0_1(hu_c), dst_affine


def gaussian_window_weights(patch_size, sigma_fraction: float = 1 / 8) -> np.ndarray:
    """Weight map for one window: 1 in the centre, falling off towards the
    border (as in nnU-Net, sigma = 1/8 of the patch side)."""
    grids = np.meshgrid(*[np.arange(s) - (s - 1) / 2 for s in patch_size], indexing="ij")
    w = np.exp(-sum((g / (s * sigma_fraction)) ** 2 / 2 for g, s in zip(grids, patch_size)))
    w /= w.max()
    return np.maximum(w, 1e-3).astype(np.float32)   # never exactly 0, so every voxel gets a weight


def window_starts(size: int, patch: int, overlap: float) -> list[int]:
    """Start positions along one axis so that windows cover [0, size) with the
    requested overlap; the last window is aligned to the end."""
    if size <= patch:
        return [0]
    step = max(1, int(patch * (1 - overlap)))
    starts = list(range(0, size - patch + 1, step))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def sliding_window_probabilities(net: torch.nn.Module, image: np.ndarray, patch_size, device,
                                 overlap: float = 0.5, batch_size: int = 2) -> np.ndarray:
    """Class probabilities (C, X, Y, Z) for the whole image. Images smaller than
    the patch are padded with 0 (= air after normalisation)."""
    pad = [(0, max(0, p - s)) for s, p in zip(image.shape, patch_size)]
    img = np.pad(image, pad, constant_values=0.0)
    weights = gaussian_window_weights(patch_size)
    w_t = torch.from_numpy(weights).to(device)
    probs = None
    norm = np.zeros(img.shape, np.float32)
    corners = list(itertools.product(*[window_starts(s, p, overlap) for s, p in zip(img.shape, patch_size)]))
    use_amp = torch.device(device).type == "cuda"

    for i in range(0, len(corners), batch_size):
        chunk = corners[i:i + batch_size]
        x = np.stack([img[c[0]:c[0] + patch_size[0], c[1]:c[1] + patch_size[1], c[2]:c[2] + patch_size[2]]
                      for c in chunk])[:, None]
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
            logits = net(torch.from_numpy(x).to(device))
        p = (torch.softmax(logits.float(), dim=1) * w_t).cpu().numpy()
        if probs is None:
            probs = np.zeros((p.shape[1],) + img.shape, np.float32)
        for c, pc in zip(chunk, p):
            sl = tuple(slice(a, a + s) for a, s in zip(c, patch_size))
            probs[(slice(None),) + sl] += pc
            norm[sl] += weights
    probs /= norm
    return probs[(slice(None),) + tuple(slice(0, s) for s in image.shape)]


def keep_largest_piece_per_class(label: np.ndarray) -> np.ndarray:
    """Remove every connected piece of a class except the largest one."""
    out = label.copy()
    for cls in np.unique(label):
        if cls == 0:
            continue
        pieces, n = ndimage.label(label == cls)
        if n > 1:
            sizes = np.bincount(pieces.ravel())
            sizes[0] = 0
            out[(pieces > 0) & (pieces != sizes.argmax())] = 0
    return out


def predict_labels_on_model_grid(net, image: np.ndarray, patch_size, device, batch_size: int = 2) -> np.ndarray:
    """Cleaned-up label volume (uint8 canonical ids) on the canonical grid."""
    probs = sliding_window_probabilities(net, image, patch_size, device, batch_size=batch_size)
    return keep_largest_piece_per_class(probs.argmax(0).astype(np.uint8))


def predict_ct(net, img: nib.Nifti1Image, patch_size, device, spacing: float = TARGET_SPACING,
               batch_size: int = 2) -> dict:
    """Full inference for one CT. Returns the labels on the model grid (RAS, ``spacing`` mm,
    used for the centroids) and on the original CT grid (used for Dice and output)."""
    image, model_affine = ct_to_model_grid(img, spacing)
    label_model = predict_labels_on_model_grid(net, image, patch_size, device, batch_size)
    label_original = resample_onto_grid(label_model, model_affine, img.affine, img.shape[:3], order=0)
    return {"label_model_grid": label_model, "model_affine": model_affine,
            "label_original_grid": label_original.astype(np.uint8)}
