"""Step 7: pictures of typical failure cases for the report (ground truth next to prediction).

What it does
  For each chosen test case: the CT, its ground-truth labels and the saved
  prediction of step 6 are put on a 2 mm RAS display grid; the side view (sagittal
  slice through the middle of the lumbar labels, head up, front of the body
  right) is drawn twice: ground truth | prediction.

Needs: step 6 (outputs/predictions/, or outputs/predictions_<tag>/ with --tag).

Output
  report/figures/failure_cases_side_views.png

    python scripts/07_make_report_figures.py
    python scripts/07_make_report_figures.py --tag 1.5mm     # 1.5 mm model, file name gets _1.5mm
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

from lumbar.geometry import resample_onto_grid
from lumbar.inference import ct_to_model_grid
from lumbar.preprocess import merge_label_files_to_canonical
from lumbar.sources import PROJECT_ROOT, load_cases

# (source, case id, caption): one example per failure type found in step 6, per model
# (key = --tag); the two models fail on partly different scans
CASES_PER_MODEL = {
    "": [   # 2 mm model
        ("verse19_test", "sub-verse012", "VerSe leaves the cut-off\nvertebra unlabelled; model: L4"),
        ("totalseg_v3_test", "s0829", "short scan, no sacrum:\ncounted one level off"),
        ("verse19_validation", "sub-verse093", "metal implants:\ncounted one level off"),
        ("verse19_validation", "sub-verse078", "fractured vertebrae:\ncounted one level off"),
    ],
    "1.5mm": [
        ("verse19_test", "sub-verse012", "(a) cut-off vertebra,\nunlabelled in VerSe"),
        ("totalseg_v3_test", "s0829", "(b) short scan without sacrum:\nlevels shifted by one"),
        ("totalseg_v3_test", "s1248", "(c) metal near the spine:\nlevels shifted by one"),
        ("verse19_test", "sub-verse250", "(d) neck and chest scan:\nvertebrae labelled L1-L5"),
    ],
}
# background, L1 red, L2 orange, L3 yellow, L4 green, L5 blue, T12 grey, sacrum purple, L6 pink
COLORS = ["#000000", "#e6194b", "#f58231", "#ffe119", "#3cb44b", "#4363d8", "#a9a9a9", "#911eb4", "#f032e6", "#9a6324"]
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


def side_view(source: str, case_id: str, predictions_dir):
    case = next(c for c in load_cases(source) if c.case_id == case_id)
    img = case.load_ct()
    image, affine = ct_to_model_grid(img)
    gt, _ = merge_label_files_to_canonical(case, img)
    pred = np.asarray(nib.load(predictions_dir / source /
                               f"{case_id}_predicted_labels.nii.gz").dataobj)
    gt = resample_onto_grid(gt, img.affine, affine, image.shape, order=0)
    pred = resample_onto_grid(pred, img.affine, affine, image.shape, order=0)
    gt[gt == 9] = 0                                   # T13 not drawn
    lumbar = np.isin(gt, (1, 2, 3, 4, 5, 8)) | np.isin(pred, (1, 2, 3, 4, 5))
    idx = np.argwhere(lumbar)
    x = int(np.median(idx[:, 0]))
    lo = np.maximum(idx.min(0) - 15, 0)
    hi = np.minimum(idx.max(0) + 15, np.array(image.shape) - 1)
    sl = (x, slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1))
    return image[sl].T, gt[sl].T, pred[sl].T            # rows: S, columns: A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="predictions of another model, e.g. 1.5mm -> outputs/predictions_1.5mm/")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    predictions_dir = PROJECT_ROOT / "outputs" / f"predictions{suffix}"
    out = FIGURES_DIR / f"failure_cases_side_views{suffix}.png"
    cases = CASES_PER_MODEL[args.tag]
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, 2 * len(cases), figsize=(2.0 * len(cases) * 1.55, 3.0),
                             gridspec_kw={"wspace": 0.04})
    for k, (source, cid, caption) in enumerate(cases):
        im, gt, pred = side_view(source, cid, predictions_dir)
        for j, (lab, title) in enumerate([(gt, "ground truth"), (pred, "prediction")]):
            ax = axes[2 * k + j]
            ax.imshow(im, origin="lower", cmap="gray", vmin=0.3, vmax=0.9)
            ax.imshow(np.ma.masked_equal(lab, 0), origin="lower", cmap=cmap, vmin=0, vmax=len(COLORS) - 1,
                      alpha=0.5, interpolation="nearest")
            ax.set_title(f"{cid}\n{title}" if j == 0 else f"\n{title}", fontsize=7)
            ax.axis("off")
        axes[2 * k].text(1.02, -0.03, caption, transform=axes[2 * k].transAxes, ha="center", va="top", fontsize=7)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[figures] -> {out}")


if __name__ == "__main__":
    main()
