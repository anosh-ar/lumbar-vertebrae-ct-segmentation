"""Part C: centre of the vertebral body only (without pedicles, laminae and processes).

Neither dataset labels the body, so we separate it by shape. In the model's isotropic
RAS grid (2 or 1.5 mm; all sizes below are in mm and converted to voxels):
  1. Erode (shrink) the vertebra mask with a ball of radius 6 mm. Every part
     thinner than 12 mm disappears: the pedicles (about 8-10 mm wide), the
     laminae and the processes. The body (about 45 mm wide, 30 mm deep, 25 mm
     high) survives as one big block.
  2. Keep the largest remaining piece that is not clearly behind (more than 5 mm
     posterior of) the centre of the whole vertebra: that is the body. The rule
     stops a thick posterior arch from being picked when the body is fractured or
     small.
  3. Grow it back by the same 6 mm, but only inside the original mask, so the
     body gets its true outline again while the cut-off posterior parts stay away.
  4. The centre of mass of that body mask, converted to world mm (RAS).

If no piece qualifies (a small vertebra fragment, e.g. cut off at the scan
edge), we fall back to the centre of the whole mask and say so.

Why this and not "the anterior half": cutting at a fixed A-P position depends on
how the patient lies and on the vertebra's size; the erosion uses only the fact
that the posterior elements are thin and the body is thick.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import voxel_to_world_mm
from .labels import CANONICAL, TARGET_IDS

EROSION_RADIUS_MM = 6.0
POSTERIOR_TOLERANCE_MM = 5.0   # normal arch core: ~20 mm behind the centre; body core: ~10 mm in front


def ball(radius_vox: int) -> np.ndarray:
    r = np.arange(-radius_vox, radius_vox + 1)
    x, y, z = np.meshgrid(r, r, r, indexing="ij")
    return x ** 2 + y ** 2 + z ** 2 <= radius_vox ** 2


def vertebral_body_mask(vertebra: np.ndarray, spacing_mm: float) -> np.ndarray | None:
    """Body part of one vertebra mask (isotropic grid), or None if erosion removes everything."""
    r = max(1, int(round(EROSION_RADIUS_MM / spacing_mm)))
    box = ndimage.find_objects(vertebra.astype(np.uint8))[0]
    box = tuple(slice(max(0, s.start - r), s.stop + r) for s in box)   # work in a small box only
    sub = vertebra[box]
    core = ndimage.binary_erosion(sub, ball(r))
    pieces, n = ndimage.label(core)
    if n == 0:
        return None
    # the body is not behind the vertebra's centre (+y = anterior in the RAS grid):
    # pieces more than POSTERIOR_TOLERANCE_MM behind the whole-vertebra centre are
    # posterior elements. The tolerance keeps a body-only mask (posterior elements
    # removed by surgery), whose core sits right at the centre, from being rejected.
    whole_y = ndimage.center_of_mass(sub)[1]
    piece_y = np.asarray(ndimage.center_of_mass(core, pieces, range(1, n + 1)))[:, 1]
    sizes = np.bincount(pieces.ravel())[1:].astype(float)
    sizes[piece_y < whole_y - POSTERIOR_TOLERANCE_MM / spacing_mm] = 0
    if sizes.max() == 0:
        return None
    body = ndimage.binary_dilation(pieces == sizes.argmax() + 1, ball(r)) & sub
    out = np.zeros(vertebra.shape, bool)
    out[box] = body
    return out


def vertebral_body_centroids(label: np.ndarray, affine: np.ndarray, ids=TARGET_IDS) -> dict:
    """{level name: {"body_ras_mm": [x,y,z], "whole_ras_mm": [x,y,z], "method": ...}}
    for every level present in ``label`` (an isotropic RAS grid, e.g. the model grid)."""
    spacing = float(np.linalg.norm(affine[:3, 0]))
    out = {}
    for c in ids:
        m = label == c
        if not m.any():
            continue
        whole = voxel_to_world_mm(affine, ndimage.center_of_mass(m))[0]
        body = vertebral_body_mask(m, spacing)
        if body is None:
            centre, method = whole, "whole_vertebra_fallback"
        else:
            centre, method = voxel_to_world_mm(affine, ndimage.center_of_mass(body))[0], "eroded_body"
        out[CANONICAL[c]] = {"body_ras_mm": [round(float(v), 2) for v in centre],
                             "whole_ras_mm": [round(float(v), 2) for v in whole], "method": method}
    return out


def centroid_json(case_id: str, centroids: dict) -> dict:
    """The output format the assignment asks for (world mm, RAS)."""
    return {"case_id": case_id, "coordinate_system": "RAS", "unit": "mm",
            "centroids": {k: v["body_ras_mm"] for k, v in centroids.items()}}
