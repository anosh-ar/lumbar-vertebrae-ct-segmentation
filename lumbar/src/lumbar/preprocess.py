"""Part A: bring any case into the canonical space.

Canonical space: RAS-aligned axes, isotropic voxels (TARGET_SPACING = 1.5 mm by
default, the native TotalSegmentator size; 2 mm was the first model, chosen per run
with --spacing), intensities
clipped to a fixed HU window and scaled to [0, 1], labels in the canonical
ids of ``labels.py``.

The HU window is fixed rather than per-case (z-score) on purpose: HU is a
calibrated unit, and a per-case normalisation would map the same bone to
different values in a whole-body TotalSegmentator scan and a tightly cropped
VerSe scan, i.e. it would create a domain gap instead of removing one.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy import ndimage

from .cases import Case
from .geometry import (AFFINE_TOL, antialias_sigma_per_axis, has_no_value_scaling, max_affine_difference,
                       ras_grid_covering_volume, read_header_summary, resample_onto_grid)
from .labels import CANONICAL, ROI_IDS

TARGET_SPACING = 1.5                 # mm, voxel size of the deliverable model; scripts take --spacing
HU_WINDOW = (-1000.0, 1500.0)
AIR_HU = -1024.0
ROI_MARGIN_MM = 30.0


# --------------------------------------------------------------------------- intensities

def load_ct_as_hu(img: nib.Nifti1Image) -> tuple[np.ndarray, list[str]]:
    """Image in HU as float32 (NIfTI scl_slope/inter applied), plus the fixes applied."""
    hu = img.get_fdata(dtype=np.float32)
    fixes = []
    if hu.ndim == 4:
        hu = hu[..., 0]
        fixes.append("4d_squeezed")
    if not np.isfinite(hu).all():
        hu[~np.isfinite(hu)] = AIR_HU
        fixes.append("nonfinite_to_air")
    if hu_offset_is_missing(hu):
        hu -= 1024.0
        fixes.append("hu_offset_minus_1024")
    return hu, fixes


def hu_offset_is_missing(hu: np.ndarray) -> bool:
    """A real CT in HU always contains fat or air (< 0). No negative values at
    all means the data were stored unsigned without the -1024 intercept."""
    return float(hu.min()) >= 0.0


def normalise_hu_to_0_1(hu: np.ndarray) -> np.ndarray:
    lo, hi = HU_WINDOW
    return ((np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.float32)


# --------------------------------------------------------------------------- labels

def bounding_box_of_nonzero(arr: np.ndarray):
    """Slices of the bounding box of the non-zero voxels, or None if empty.
    Three axis projections are much cheaper than np.nonzero on a full volume."""
    box = []
    for ax in range(arr.ndim):
        nz = np.flatnonzero(arr.any(axis=tuple(a for a in range(arr.ndim) if a != ax)))
        if nz.size == 0:
            return None
        box.append(slice(int(nz[0]), int(nz[-1]) + 1))
    return tuple(box)


def merge_label_files_to_canonical(case: Case, ref: nib.Nifti1Image) -> tuple[np.ndarray, dict]:
    """All label files of a case as one canonical uint8 volume on the image grid.

    Handles: float-stored masks (rounded, residual reported), masks on a
    different grid than the image (nearest-neighbour resampled onto it),
    and voxels claimed by more than one mask (later mask wins, counted).
    All per-voxel work happens inside each mask's bounding box.
    """
    shape = ref.shape[:3]
    out = np.zeros(shape, np.uint8)
    report = {"parts": {}, "overlap_voxels": 0, "overlaps": {}, "raw_counts": {}}

    for part in case.label_files:
        pimg = case.load_label_file(part)
        h = read_header_summary(pimg)
        raw = np.asanyarray(pimg.dataobj)
        if raw.ndim == 4:
            raw = raw[..., 0]
        rec = {"dtype": h["dtype"], "scaled": not has_no_value_scaling(h["scl_slope"], h["scl_inter"]),
               "affine_diff": max_affine_difference(pimg.affine, ref.affine),
               "same_shape": tuple(raw.shape) == tuple(shape), "nonint_residual": 0.0,
               "resampled": False}
        report["parts"][part.name] = rec

        box = bounding_box_of_nonzero(raw)
        if box is None:
            continue
        if raw.dtype.kind == "f":
            sub = raw[box]
            rounded = np.rint(sub)
            rec["nonint_residual"] = float(np.abs(sub - rounded).max())
            raw = np.zeros(raw.shape, np.int32)
            raw[box] = rounded
        if raw.dtype.kind == "i":
            raw = np.clip(raw, 0, None)

        if not rec["same_shape"] or rec["affine_diff"] > AFFINE_TOL:
            raw = resample_onto_grid(raw, pimg.affine, ref.affine, shape, order=0)
            rec["resampled"] = True
            box = bounding_box_of_nonzero(raw)
            if box is None:
                continue

        sub = raw[box]
        sub = sub.astype(np.uint8 if sub.max() < 256 else np.int32, copy=False)
        counts = np.bincount(sub.ravel())
        lut = np.zeros(max(len(counts), max(part.value_to_canonical_id) + 1), np.uint8)
        for r, c in part.value_to_canonical_id.items():
            lut[r] = c
        for v in np.nonzero(counts)[0]:
            if v:
                report["raw_counts"][int(v)] = report["raw_counts"].get(int(v), 0) + int(counts[v])

        mapped = lut[sub]
        m = mapped > 0
        dst = out[box]                      # view into out
        clash = m & (dst > 0)
        n = int(clash.sum())
        if n:
            report["overlap_voxels"] += n
            prev_vals = dst[clash]
            for prev in np.unique(prev_vals):
                key = f"{CANONICAL[int(prev)]}&{part.name}"
                report["overlaps"][key] = report["overlaps"].get(key, 0) + int((prev_vals == prev).sum())
        dst[m] = mapped[m]
    return out, report


# --------------------------------------------------------------------------- full case

def lumbar_region_bounds(label: np.ndarray, zooms, margin_mm: float = ROI_MARGIN_MM):
    """Inclusive voxel bounds of all ROI structures plus a margin, or None."""
    objs = ndimage.find_objects(label)
    boxes = [objs[i - 1] for i in ROI_IDS if i - 1 < len(objs) and objs[i - 1] is not None]
    if not boxes:
        return None
    lo = np.min([[s.start for s in b] for b in boxes], axis=0)
    hi = np.max([[s.stop - 1 for s in b] for b in boxes], axis=0)
    pad = np.ceil(margin_mm / np.asarray(zooms)).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.asarray(label.shape) - 1)
    return lo, hi


def preprocess_case_to_canonical(case: Case, spacing: float = TARGET_SPACING, crop: str = "lumbar") -> dict | None:
    """Canonical-space image/label for one case. Returns None when ``crop`` is
    'lumbar' and the case contains none of the ROI structures."""
    img = case.load_ct()
    hu, fixes = load_ct_as_hu(img)
    label, lrep = merge_label_files_to_canonical(case, img)
    affine = img.affine.copy()
    zooms = np.linalg.norm(affine[:3, :3], axis=0)

    bounds = None
    if crop == "lumbar":
        bounds = lumbar_region_bounds(label, zooms)
        if bounds is None:
            return None
        lo, hi = bounds
        sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
        hu, label = hu[sl], label[sl]
        shift = np.eye(4)
        shift[:3, 3] = lo
        affine = affine @ shift

    sigma = antialias_sigma_per_axis(affine, spacing)
    if (sigma > 0).any():
        hu = ndimage.gaussian_filter(hu, sigma, mode="nearest")

    dst_affine, dst_shape = ras_grid_covering_volume(affine, hu.shape, spacing)
    hu_c = resample_onto_grid(hu, affine, dst_affine, dst_shape, order=1, cval=AIR_HU)
    lab_c = resample_onto_grid(label, affine, dst_affine, dst_shape, order=0, cval=0).astype(np.uint8)

    present = {CANONICAL[int(i)]: int((lab_c == i).sum()) for i in np.unique(lab_c) if i}
    meta = {
        "case_id": case.case_id, "dataset": case.dataset, "split": case.split,
        "subject_id": case.subject_id, "image_path": case.image_path,
        "source": read_header_summary(img), "source_affine": img.affine.tolist(),
        "crop": crop, "crop_bounds_vox": None if bounds is None else [bounds[0].tolist(), bounds[1].tolist()],
        "spacing_mm": spacing, "hu_window": list(HU_WINDOW), "antialias_sigma_vox": sigma.tolist(),
        "intensity_fixes": fixes, "label_report": lrep, "voxels_per_label": present,
        "affine": dst_affine.tolist(), "shape": list(dst_shape),
    }
    return {"image": normalise_hu_to_0_1(hu_c), "label": lab_c, "affine": dst_affine, "meta": meta}
