"""Step 9: picture of how the vertebral body centre is found (Part C), for the report.

What it does
  Takes one VerSe vertebra (ground-truth mask, 1.5 mm RAS grid, same code as
  training) and draws one slice seen from above (front of the body up) after
  each step of the body-centre method:
    1. the vertebra mask,
    2. the mask shrunk by 6 mm: thin parts (arch, processes) vanish,
    3. the body grown back inside the mask, with three centres:
       whole vertebra, the body centre, and the VerSe annotated point.
  Needs no GPU and no torch: only the downloaded VerSe data.

Output
  report/figures/body_centre_steps.png

    python scripts/09_draw_body_centre_steps.py [--case sub-verse020] [--level L3]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

from lumbar.body_centroid import EROSION_RADIUS_MM, ball, vertebral_body_mask
from lumbar.geometry import affine_after_reorienting, voxel_to_world_mm
from lumbar.labels import CANONICAL, VERSE_LABELS
from lumbar.preprocess import preprocess_case_to_canonical
from lumbar.sources import PROJECT_ROOT, load_cases

OUT = PROJECT_ROOT / "report" / "figures" / "body_centre_steps.png"


def verse_point_in_grid(case, level: str, grid_affine: np.ndarray):
    """VerSe annotated point of ``level`` as a voxel position in the model grid, or None."""
    img = case.load_ct()
    ctd = case.load_verse_centroids()
    if not ctd or not ctd["direction"]:
        return None
    aff = affine_after_reorienting(img.affine, img.shape[:3], ctd["direction"])
    for raw, xyz in ctd["points"].items():
        if CANONICAL.get(VERSE_LABELS.get(raw, 0)) == level:
            world = voxel_to_world_mm(aff, xyz)[0]
            return (np.linalg.inv(grid_affine) @ np.append(world, 1))[:3]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="verse19_test")
    ap.add_argument("--case", default="sub-verse020")
    ap.add_argument("--level", default="L3")
    args = ap.parse_args()

    case = next(c for c in load_cases(args.source) if c.case_id == args.case)
    res = preprocess_case_to_canonical(case)
    image, label, affine = res["image"], res["label"], res["affine"]
    level_id = next(i for i, n in CANONICAL.items() if n == args.level)
    vertebra = label == level_id
    spacing = float(np.linalg.norm(affine[:3, 0]))

    r = max(1, int(round(EROSION_RADIUS_MM / spacing)))
    shrunk = ndimage.binary_erosion(vertebra, ball(r))
    body = vertebral_body_mask(vertebra, spacing)
    whole_c = np.array(ndimage.center_of_mass(vertebra))
    body_c = np.array(ndimage.center_of_mass(body))
    verse_c = verse_point_in_grid(case, args.level, affine)

    # axial slice through the body centre; array axes are (R, A, S)
    z = int(round(body_c[2]))
    xs, ys = np.nonzero(vertebra[:, :, z])
    pad = 8
    sx = slice(max(xs.min() - pad, 0), xs.max() + pad + 1)
    sy = slice(max(ys.min() - pad, 0), ys.max() + pad + 1)

    def crop(a):      # rows = anterior up, columns = patient's right on the left (radiology view)
        return a[sx, sy, z].T[:, ::-1]

    def to_plot(c):   # voxel position -> (column, row) in the cropped picture
        return sx.stop - 1 - c[0], c[1] - sy.start

    # 0 nothing, 1 vertebra / kept, 2 removed
    step1 = crop(vertebra).astype(int)
    step2 = np.where(crop(shrunk), 1, np.where(crop(vertebra), 2, 0))
    step3 = np.where(crop(body), 1, np.where(crop(vertebra), 2, 0))
    cmap = ListedColormap(["#000000", "#3cb44b", "#e6194b"])
    titles = [f"1. vertebra mask\n({args.case} {args.level}, seen from above)",
              f"2. shrink by {EROSION_RADIUS_MM:.0f} mm:\nthin parts vanish (red)",
              "3. grow the body back\ninside the mask; take its centre"]

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5), gridspec_kw={"wspace": 0.05})
    for ax, lab, title in zip(axes, [step1, step2, step3], titles):
        ax.imshow(crop(image), origin="lower", cmap="gray", vmin=0.3, vmax=0.9)
        ax.imshow(np.ma.masked_equal(lab, 0), origin="lower", cmap=cmap, vmin=0, vmax=2,
                  alpha=0.55, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    ax = axes[2]
    ax.plot(*to_plot(whole_c), "x", color="white", ms=7, mew=2, label="whole-vertebra centre")
    ax.plot(*to_plot(body_c), "o", color="#ffe119", ms=6, mec="black", label="body centre (this method)")
    if verse_c is not None:
        ax.plot(*to_plot(verse_c), "+", color="#00e5ff", ms=9, mew=2, label="VerSe annotated point")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, facecolor="#333333",
              labelcolor="white", edgecolor="none")
    axes[0].text(0.5, 0.98, "front", transform=axes[0].transAxes, ha="center", va="top",
                 color="white", fontsize=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    print(f"[figures] -> {OUT}")


if __name__ == "__main__":
    main()
