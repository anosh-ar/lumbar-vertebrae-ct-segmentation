"""Step 10: illustration for the README: what the model's output looks like on good test scans.

What it does
  For a few test scans that the model segments well, it draws the side view (sagittal
  slice through the middle of the predicted lumbar spine, head up, front of the body
  right) with the predicted vertebrae L1-L5 coloured and the predicted vertebral body
  centre of each level marked with a white cross. The centres are those of the step-6
  centroid JSON files, projected onto the slice.

Needs: step 6 for the 1.5 mm model (outputs/predictions_1.5mm/).

Output
  report/figures/example_segmentations.png

    python scripts/10_draw_example_segmentations.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

from lumbar.geometry import resample_onto_grid
from lumbar.inference import ct_to_model_grid
from lumbar.labels import TARGET_IDS
from lumbar.sources import PROJECT_ROOT, load_cases

# well-segmented, complete scans (all of L1-L5, none cut off), two per dataset
CASES = [("totalseg_v3_test", "s1377", "TotalSegmentator"), ("totalseg_v3_test", "s1322", "TotalSegmentator"),
         ("verse19_test", "sub-verse029", "VerSe"), ("verse19_validation", "sub-verse269", "VerSe")]
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions_1.5mm"
# background, L1 red, L2 orange, L3 yellow, L4 green, L5 blue
COLORS = ["#000000", "#e6194b", "#f58231", "#ffe119", "#3cb44b", "#4363d8"]
OUT = PROJECT_ROOT / "report" / "figures" / "example_segmentations.png"


def side_view_with_centres(source: str, case_id: str):
    """(CT slice, label slice, centre points (column, row)) of the sagittal mid-slice."""
    case = next(c for c in load_cases(source) if c.case_id == case_id)
    img = case.load_ct()
    image, affine = ct_to_model_grid(img)
    pred = np.asarray(nib.load(PREDICTIONS_DIR / source / f"{case_id}_predicted_labels.nii.gz").dataobj)
    pred = resample_onto_grid(pred, img.affine, affine, image.shape, order=0)
    pred[~np.isin(pred, TARGET_IDS)] = 0                  # show L1-L5 only
    idx = np.argwhere(pred > 0)
    x = int(np.median(idx[:, 0]))
    lo = np.maximum(idx.min(0) - 20, 0)
    hi = np.minimum(idx.max(0) + 20, np.array(image.shape) - 1)
    centres = json.loads((PREDICTIONS_DIR / source / f"{case_id}_centroids.json").read_text())["centroids"]
    to_voxel = np.linalg.inv(affine)
    points = []
    for xyz in centres.values():
        v = to_voxel @ np.array([*xyz, 1.0])
        points.append((v[1] - lo[1], v[2] - lo[2]))          # (anterior column, superior row) in the crop
    sl = (x, slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1))
    return image[sl].T, pred[sl].T, points


def main():
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, len(CASES), figsize=(2.3 * len(CASES), 3.6), gridspec_kw={"wspace": 0.05})
    for ax, (source, cid, dataset) in zip(axes, CASES):
        im, lab, points = side_view_with_centres(source, cid)
        ax.imshow(im, origin="lower", cmap="gray", vmin=0.3, vmax=0.9)
        ax.imshow(np.ma.masked_equal(lab, 0), origin="lower", cmap=cmap, vmin=0, vmax=len(COLORS) - 1,
                  alpha=0.5, interpolation="nearest")
        for col, row in points:
            ax.plot(col, row, marker="x", color="white", markersize=6, markeredgewidth=1.6)
        ax.set_title(f"{dataset}\n{cid}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Predicted L1 (red) to L5 (blue); white cross: predicted vertebral body centre", fontsize=8, y=0.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"[figures] -> {OUT}")


if __name__ == "__main__":
    main()
