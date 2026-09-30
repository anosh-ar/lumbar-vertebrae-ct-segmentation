"""Header inspection, coordinate transforms and resampling.

NIfTI world coordinates are RAS+ in mm: +x toward the patient's right,
+y anterior, +z superior. ``img.affine`` maps voxel indices to that space.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, inv_ornt_aff, io_orientation, ornt_transform
from scipy import ndimage

AFFINE_TOL = 1e-3   # max abs difference (mm / direction cosine) for "same grid"


def read_header_summary(img: nib.Nifti1Image) -> dict:
    """Geometry and scaling fields that the audit checks."""
    hdr = img.header
    zooms = np.asarray(hdr.get_zooms()[:3], dtype=float)
    q, q_code = hdr.get_qform(coded=True)
    s, s_code = hdr.get_sform(coded=True)
    qs_diff = float(np.abs(q - s).max()) if (q is not None and s is not None) else np.nan

    direction = img.affine[:3, :3] / np.linalg.norm(img.affine[:3, :3], axis=0)
    shear = float(np.abs(direction.T @ direction - np.eye(3)).max())

    return {
        "shape": tuple(int(x) for x in img.shape[:3]),
        "ndim": len(img.shape),
        "dtype": str(hdr.get_data_dtype()),
        "zooms": tuple(round(float(z), 4) for z in zooms),
        "axcodes": "".join(nib.aff2axcodes(img.affine)),
        "qform_code": int(q_code),
        "sform_code": int(s_code),
        "qs_maxdiff": qs_diff,
        "shear": shear,
        "det_sign": int(np.sign(np.linalg.det(img.affine[:3, :3]))),
        "scl_slope": float(hdr["scl_slope"]),
        "scl_inter": float(hdr["scl_inter"]),
    }


def has_no_value_scaling(slope: float, inter: float) -> bool:
    """NIfTI treats slope 0 or NaN as 'no scaling'."""
    return (not np.isfinite(slope) or slope in (0.0, 1.0)) and (not np.isfinite(inter) or inter == 0.0)


def max_affine_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a[:3, :] - b[:3, :]).max())


def voxel_to_world_mm(affine: np.ndarray, ijk) -> np.ndarray:
    ijk = np.atleast_2d(np.asarray(ijk, dtype=float))
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def affine_after_reorienting(affine: np.ndarray, shape, axcodes) -> np.ndarray:
    """Affine of the same image after reorienting its array to ``axcodes``
    (e.g. ("P","I","R")). Used to interpret VerSe centroids, which are voxel
    indices in such a reoriented array."""
    xform = ornt_transform(io_orientation(affine), axcodes2ornt(tuple(axcodes)))
    return affine @ inv_ornt_aff(xform, shape)


def resample_onto_grid(arr: np.ndarray, src_affine: np.ndarray, dst_affine: np.ndarray,
                       dst_shape, order: int, cval: float = 0.0) -> np.ndarray:
    """Pull ``arr`` onto the destination voxel grid. Handles any combination of
    axis permutation, flips, spacing and shear (gantry tilt) in one step,
    because it works purely through the voxel->world->voxel mapping."""
    m = np.linalg.inv(src_affine) @ dst_affine
    return ndimage.affine_transform(arr, m[:3, :3], offset=m[:3, 3], output_shape=tuple(dst_shape),
                                    order=order, mode="constant", cval=cval, prefilter=False)


def ras_grid_covering_volume(src_affine: np.ndarray, src_shape, spacing: float,
                             ijk_min=None, ijk_max=None) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Axis-aligned RAS grid with isotropic ``spacing`` covering the world-space
    bounding box of a (sub)volume of the source, given as inclusive voxel bounds."""
    lo = np.zeros(3) if ijk_min is None else np.asarray(ijk_min, float)
    hi = np.asarray(src_shape[:3], float) - 1 if ijk_max is None else np.asarray(ijk_max, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = voxel_to_world_mm(src_affine, corners)
    wmin, wmax = world.min(0), world.max(0)
    shape = tuple(int(np.floor((wmax[k] - wmin[k]) / spacing)) + 1 for k in range(3))
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = wmin
    return affine, shape


def antialias_sigma_per_axis(src_affine: np.ndarray, spacing: float) -> np.ndarray:
    """Gaussian sigma per source axis (in voxels) before downsampling to
    ``spacing``: (factor - 1) / 2 where the axis is downsampled, else 0."""
    zooms = np.linalg.norm(src_affine[:3, :3], axis=0)
    factor = spacing / zooms
    return np.where(factor > 1, (factor - 1) / 2, 0.0)
