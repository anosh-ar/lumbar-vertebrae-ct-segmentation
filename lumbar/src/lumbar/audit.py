"""Part A: find header, intensity and label anomalies in one case.

``audit_case_for_anomalies`` returns one row per case: raw measurements plus
the anomaly codes found, space-separated in ``row["flags"]``.
``ANOMALIES`` explains every code in plain words (the problem, how it is
found, what the pipeline does about it); the table in the anomaly report is
generated from it, so code and report cannot drift apart.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
from scipy import ndimage

from .cases import Case
from .geometry import (AFFINE_TOL, affine_after_reorienting, has_no_value_scaling, read_header_summary,
                       voxel_to_world_mm)
from .labels import CANONICAL, TARGET_LEVELS, VERSE_LABELS, VERSE_NAMES
from .preprocess import HU_WINDOW, ROI_MARGIN_MM, load_ct_as_hu, lumbar_region_bounds, merge_label_files_to_canonical

MIN_VERT_ML, MAX_VERT_ML = 10.0, 150.0   # plausible adult lumbar vertebra incl. posterior elements
STRAY_ML = 0.5                           # extra connected components above this are flagged
FRAGMENT_ML = 2.0                        # a 'level' smaller than this is a label fragment
METAL_HU, METAL_ML = 3000.0, 0.5
PAD_HU, PAD_FRACTION = -2000.0, 0.01
SHEAR_TOL = 1e-3
ANISO_RATIO = 3.0


class Anomaly(NamedTuple):
    problem: str        # what is wrong, in plain words
    how_found: str      # the exact test the audit runs
    what_we_do: str     # what the pipeline does about it today ("Plan:" = not implemented yet)


ANOMALIES = {
    # ---- file and header
    "LOAD_ERROR": Anomaly(
        "The file cannot be opened",
        "Reading the file raises an error",
        "Case is skipped by every step; the error text is saved in the CSV"),
    "NO_XFORM": Anomaly(
        "The header has no position or orientation (qform and sform both unset)",
        "qform_code and sform_code are both 0",
        "Flagged only; such a case would need a manual check before use"),
    "QS_MISMATCH": Anomaly(
        "The header's two position/orientation fields (qform and sform) disagree",
        f"Both are set and differ by more than {AFFINE_TOL} mm",
        "We use the sform, as the NIfTI standard and nibabel do"),
    "BAD_PIXDIM": Anomaly(
        "Voxel size is zero, negative or missing",
        "A voxel size is <= 0 or not a number",
        "Flagged only; such a case would need a manual check before use"),
    "SHEAR": Anomaly(
        "Voxel axes are not at right angles (e.g. tilted CT gantry)",
        f"Axis directions are off from perpendicular by more than {SHEAR_TOL}",
        "Corrected when we resample onto the RAS grid (the resampling uses the full affine)"),
    "NONAXIAL": Anomaly(
        "Slices are not stored as axial slices (e.g. stored as sagittal slices)",
        "The third array axis does not run head-to-foot",
        "Rotated into the standard RAS orientation when we resample"),
    "ANISO": Anomaly(
        "Voxels are much longer in one direction than the others (thick slices)",
        f"Longest voxel side is more than {ANISO_RATIO:g}x the shortest",
        "Smoothed before downsampling so no aliasing appears; flagged because training data has thin slices"),
    # ---- image intensities
    "IMG_4D": Anomaly(
        "Image is stored with an extra, empty 4th dimension",
        "The image array has 4 dimensions",
        "The 4th dimension is dropped"),
    "IMG_SCALED": Anomaly(
        "Stored values must be rescaled to get HU (scl_slope / scl_inter)",
        "Header slope is not 1 or intercept is not 0",
        "Rescaled when loading, as the NIfTI standard says; all HU checks use the rescaled values"),
    "HU_OFFSET": Anomaly(
        "Intensities are not in Hounsfield units: there are no negative values",
        "Smallest value is >= 0 (a real CT always contains air or fat, which are negative)",
        "1024 is subtracted to get HU"),
    "NONFINITE": Anomaly(
        "Some voxels are NaN or infinite",
        "A voxel value is not a finite number",
        "Set to air (-1024 HU)"),
    "PAD_OUTSIDE_FOV": Anomaly(
        "The area outside the scanned circle is filled with a very low value",
        f"More than {PAD_FRACTION:.0%} of voxels are below {PAD_HU:.0f} HU (air is about -1000)",
        f"Clipped to {HU_WINDOW[0]:.0f} HU by the intensity window, so it looks like air"),
    "METAL": Anomaly(
        "Metal near the lumbar spine (e.g. screws or implants)",
        f"More than {METAL_ML} ml above {METAL_HU:.0f} HU within the labelled vertebrae + {ROI_MARGIN_MM:.0f} mm",
        f"Clipped to {HU_WINDOW[1]:.0f} HU by the intensity window; flagged to explain errors later"),
    # ---- label files
    "LABEL_GRID_MISMATCH": Anomaly(
        "The label mask does not line up with the CT (other size or position)",
        f"Array shape differs, or position/orientation differs by more than {AFFINE_TOL} mm",
        "Mask is resampled onto the CT grid (nearest neighbour, so labels stay whole numbers)"),
    "LABEL_AFFINE_EPS": Anomaly(
        "The label's position differs from the CT's only by rounding noise",
        f"Position/orientation differs by more than 0 but at most {AFFINE_TOL} mm",
        "Treated as the same grid"),
    "LABEL_FLOAT": Anomaly(
        "The label mask is stored as decimal numbers instead of integers",
        "Stored data type is float",
        "Rounded to whole numbers"),
    "LABEL_NONINT": Anomaly(
        "The label mask contains values that are not whole numbers",
        "A value is more than 0.001 away from the nearest whole number",
        "Rounded; the largest deviation is saved in the CSV"),
    "LABEL_SCALED": Anomaly(
        "Label values must be rescaled (scl_slope / scl_inter)",
        "Header slope is not 1 or intercept is not 0",
        "Rescaled when loading, then rounded"),
    "MASK_OVERLAP": Anomaly(
        "A voxel belongs to two TotalSegmentator masks at the same time",
        "The voxel is non-zero in two of the per-structure mask files",
        "The lumbar vertebra wins over T12/sacrum (lumbar masks are merged last); overlap size saved in the CSV"),
    # ---- anatomy in the labels
    "NO_LUMBAR": Anomaly(
        "No lumbar vertebra is labelled (the scan does not cover the lumbar spine)",
        "None of L1-L5 is in the mask",
        "Not usable for L1-L5. Still preprocessed if T12 or sacrum is present, "
        "so the training-case selection must filter it out"),
    "LEVEL_GAP": Anomaly(
        "A lumbar level is missing between two present ones (e.g. L2 and L4 but no L3)",
        "The present lumbar levels are not consecutive",
        "Flagged. Plan (Part D): the missing level is not scored"),
    "TRUNCATED": Anomaly(
        "A lumbar vertebra is cut off at the edge of the scan",
        "Its mask touches the image border",
        "Kept for training; flagged because Dice on a partial vertebra is not comparable"),
    "MULTI_CC": Anomaly(
        "A vertebra label is split into separate pieces",
        f"Pieces other than the largest add up to more than {STRAY_ML} ml",
        "Left as it is; reported"),
    "FRAGMENT": Anomaly(
        "A lumbar level is only a tiny scrap of label",
        f"The level's volume is below {FRAGMENT_ML:g} ml (a real lumbar vertebra is about 60 ml)",
        "Ignored in the order check. Plan (Part D): not scored as a vertebra"),
    "VOLUME_OUTLIER": Anomaly(
        "A vertebra has an implausible volume",
        f"A lumbar vertebra that is not cut off measures {FRAGMENT_ML:g}-{MIN_VERT_ML:g} ml or more than {MAX_VERT_ML:g} ml",
        "Flagged for a visual check"),
    "ORDER_VIOLATION": Anomaly(
        "Vertebrae are in the wrong top-to-bottom order (e.g. L3 above L2)",
        "Centres of T12, L1-L5 and sacrum do not go strictly downwards (tiny scraps ignored)",
        "Flagged: probably a label mix-up in the ground truth; to be checked visually"),
    "TRANSITIONAL_L6": Anomaly(
        "The patient has a sixth lumbar vertebra (VerSe label 25)",
        "Label 25 is in the mask",
        "Kept as its own class (id 8), never merged into L5 or background. Plan (Part D): reported separately"),
    "TRANSITIONAL_T13": Anomaly(
        "The patient has a thirteenth thoracic vertebra (VerSe label 28)",
        "Label 28 is in the mask",
        "Kept as its own class (id 9), never merged into T12 or background. Plan (Part D): reported separately"),
    # ---- dataset structure
    "SPLIT_SCAN": Anomaly(
        "One patient's spine is split over several scans",
        "The VerSe file name contains '_split-'",
        "Each scan is processed on its own; the patient id is kept so results can be grouped per patient"),
    "CTD_LABEL_MISMATCH": Anomaly(
        "The VerSe centroid file and the mask list different lumbar levels",
        "The lumbar labels (L1-L6) in the two files differ",
        "Reported; only levels present in both are compared"),
}


def _vertebrae_out_of_order(centroids: dict[str, np.ndarray]) -> bool:
    seq = ["T13", "T12", "L1", "L2", "L3", "L4", "L5", "L6", "sacrum"]
    z = [centroids[n][2] for n in seq if n in centroids]
    return any(a <= b for a, b in zip(z, z[1:]))


def audit_case_for_anomalies(case: Case) -> dict:
    row = {"case_id": case.case_id, "dataset": case.dataset, "split": case.split,
           "subject_id": case.subject_id}
    flags: list[str] = []
    try:
        img = case.load_ct()
    except Exception as e:  # noqa: BLE001 - any read failure is an audit finding
        row.update(flags="LOAD_ERROR", error=repr(e))
        return row

    h = read_header_summary(img)
    zooms = np.asarray(h["zooms"])
    row.update({f"img_{k}": v for k, v in h.items()})
    row["img_aniso"] = float(zooms.max() / zooms.min()) if zooms.min() > 0 else np.nan

    if h["qform_code"] == 0 and h["sform_code"] == 0:
        flags.append("NO_XFORM")
    if h["qform_code"] and h["sform_code"] and h["qs_maxdiff"] > AFFINE_TOL:
        flags.append("QS_MISMATCH")
    if not np.all(np.isfinite(zooms)) or (zooms <= 0).any():
        flags.append("BAD_PIXDIM")
    if h["shear"] > SHEAR_TOL:
        flags.append("SHEAR")
    if h["axcodes"][2] not in "SI":
        flags.append("NONAXIAL")
    if row["img_aniso"] > ANISO_RATIO:
        flags.append("ANISO")
    if h["ndim"] == 4:
        flags.append("IMG_4D")
    if not has_no_value_scaling(h["scl_slope"], h["scl_inter"]):
        flags.append("IMG_SCALED")

    # ---- intensities
    hu, fixes = load_ct_as_hu(img)
    voxel_ml = float(np.abs(np.linalg.det(img.affine[:3, :3]))) / 1000.0
    sub = hu[::2, ::2, ::2]
    p = np.percentile(sub, [0.1, 1, 50, 99, 99.9])
    row.update(hu_min=float(hu.min()), hu_p01=p[0], hu_p1=p[1], hu_p50=p[2], hu_p99=p[3],
               hu_p999=p[4], hu_max=float(hu.max()), intensity_fixes=";".join(fixes))
    row["metal_ml"] = float((hu > METAL_HU).sum()) * voxel_ml
    row["pad_fraction"] = float((sub < PAD_HU).mean())
    del sub
    if "hu_offset_minus_1024" in fixes:
        flags.append("HU_OFFSET")
    if "nonfinite_to_air" in fixes:
        flags.append("NONFINITE")
    if row["pad_fraction"] > PAD_FRACTION:
        flags.append("PAD_OUTSIDE_FOV")

    # ---- labels
    label, lrep = merge_label_files_to_canonical(case, img)
    bounds = lumbar_region_bounds(label, zooms)
    if bounds is not None:
        roi = tuple(slice(a, b + 1) for a, b in zip(*bounds))
        row["metal_roi_ml"] = float((hu[roi] > METAL_HU).sum()) * voxel_ml
        if row["metal_roi_ml"] > METAL_ML:
            flags.append("METAL")
    del hu
    parts = list(lrep["parts"].values())
    row["label_dtypes"] = ";".join(sorted({r["dtype"] for r in parts}))
    row["label_max_affine_diff"] = max(r["affine_diff"] for r in parts) if parts else np.nan
    row["label_nonint_residual"] = max(r["nonint_residual"] for r in parts) if parts else np.nan
    row["overlap_voxels"] = lrep["overlap_voxels"]
    row["overlaps"] = json.dumps(lrep["overlaps"]) if lrep["overlaps"] else ""
    if any(r["resampled"] for r in parts):
        flags.append("LABEL_GRID_MISMATCH")
    elif any(0 < r["affine_diff"] <= AFFINE_TOL for r in parts):
        flags.append("LABEL_AFFINE_EPS")
    if any(r["dtype"].startswith("float") for r in parts):
        flags.append("LABEL_FLOAT")
    if row["label_nonint_residual"] > 1e-3:
        flags.append("LABEL_NONINT")
    if any(r["scaled"] for r in parts):
        flags.append("LABEL_SCALED")
    if lrep["overlap_voxels"]:
        flags.append("MASK_OVERLAP")
    if case.dataset == "verse":
        row["verse_labels"] = " ".join(VERSE_NAMES.get(v, str(v)) for v in sorted(lrep["raw_counts"]))

    # ---- per-structure geometry
    objs = ndimage.find_objects(label)
    centroids: dict[str, np.ndarray] = {}
    truncated, multi, outlier, fragments = [], [], [], []
    for cid, name in CANONICAL.items():
        sl = objs[cid - 1] if cid - 1 < len(objs) else None
        if sl is None:
            continue
        m = label[sl] == cid
        ml = float(m.sum()) * voxel_ml
        touches = any(s.start == 0 or s.stop == n for s, n in zip(sl, label.shape))
        cc, n_cc = ndimage.label(m, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(cc.ravel())[1:]
        extra_ml = float(sizes.sum() - sizes.max()) * voxel_ml
        com = np.asarray(ndimage.center_of_mass(m)) + [s.start for s in sl]
        centroids[name] = voxel_to_world_mm(img.affine, com)[0]
        row[f"{name}_ml"] = round(ml, 2)
        if name in TARGET_LEVELS:
            row[f"{name}_ncc"] = int(n_cc)
            if touches:
                truncated.append(name)
            if ml < FRAGMENT_ML:
                fragments.append(name)
            elif not touches and not (MIN_VERT_ML <= ml <= MAX_VERT_ML):
                outlier.append(name)
        if extra_ml > STRAY_ML:
            multi.append(name)

    present = [n for n in TARGET_LEVELS if n in centroids]
    row["lumbar_present"] = " ".join(present)
    row["n_lumbar"] = len(present)
    row["truncated"] = " ".join(truncated)
    row["multi_cc"] = " ".join(multi)
    row["volume_outlier"] = " ".join(outlier)
    row["fragments"] = " ".join(fragments)
    row["centroids_ras"] = json.dumps({k: [round(float(x), 1) for x in v] for k, v in centroids.items()})
    if not present:
        flags.append("NO_LUMBAR")
    else:
        idx = [TARGET_LEVELS.index(n) for n in present]
        if idx != list(range(idx[0], idx[-1] + 1)):
            flags.append("LEVEL_GAP")
    if truncated:
        flags.append("TRUNCATED")
    if multi:
        flags.append("MULTI_CC")
    if outlier:
        flags.append("VOLUME_OUTLIER")
    if fragments:
        flags.append("FRAGMENT")
    if _vertebrae_out_of_order({k: v for k, v in centroids.items() if k not in fragments}):
        flags.append("ORDER_VIOLATION")
    if "L6" in centroids:
        flags.append("TRANSITIONAL_L6")
    if "T13" in centroids:
        flags.append("TRANSITIONAL_T13")
    if "_split-" in case.case_id:
        flags.append("SPLIT_SCAN")

    # ---- VerSe centroid annotations vs. mask centroids
    ctd = case.load_verse_centroids()
    if ctd and ctd["direction"]:
        row["ctd_direction"] = "".join(ctd["direction"])
        aff = affine_after_reorienting(img.affine, img.shape[:3], ctd["direction"])
        zr = np.linalg.norm(aff[:3, :3], axis=0)
        d_vox, d_mm, ant = [], [], []
        for raw_label, xyz in ctd["points"].items():
            name = CANONICAL.get(VERSE_LABELS.get(raw_label, 0))
            if name not in centroids:
                continue
            w_vox = voxel_to_world_mm(aff, xyz)[0]                   # hypothesis A: voxel indices
            w_mm = voxel_to_world_mm(aff, np.asarray(xyz) / zr)[0]   # hypothesis B: mm along the axes
            d_vox.append(np.linalg.norm(w_vox - centroids[name]))
            d_mm.append(np.linalg.norm(w_mm - centroids[name]))
            ant.append(w_vox[1] - centroids[name][1])                # +y = anterior in RAS
        row["ctd_n_compared"] = len(d_vox)
        if d_vox:
            row["ctd_dist_med_vox_hyp"] = float(np.median(d_vox))
            row["ctd_dist_med_mm_hyp"] = float(np.median(d_mm))
            row["ctd_anterior_offset_med"] = float(np.median(ant))
        ctd_l = {VERSE_NAMES[k] for k in ctd["points"] if 20 <= k <= 25}
        msk_l = {VERSE_NAMES[k] for k in lrep["raw_counts"] if 20 <= k <= 25}
        if ctd_l != msk_l:
            flags.append("CTD_LABEL_MISMATCH")
            row["ctd_mismatch"] = f"ctd-only:{sorted(ctd_l - msk_l)} mask-only:{sorted(msk_l - ctd_l)}"

    row["flags"] = " ".join(flags)
    return row
