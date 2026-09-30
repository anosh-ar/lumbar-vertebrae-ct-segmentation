"""Step 3 (Part A): prove that preprocessing kept every vertebra in the right place.

What it does
  1. Numbers: for every structure (L1, L2, ..., T12, sacrum) it compares the
     centre point in world mm and the volume after preprocessing with the same
     values measured on the original file in step 1. Correct rotation and
     resampling agree to about 1 mm; a flipped or swapped axis is off by
     centimetres.
  2. Pictures: one side view (mid-sagittal slice) per case with the labels on
     top, all drawn the same way: head up, belly to the right. A spine drawn
     the wrong way round means a rotation bug.

Needs: the outputs of step 1 (anomalies_per_case_<source>.csv) and step 2.

Output (outputs/qc/)
  preprocessing_check_<folder>.csv                 one row per case: worst centre error, volume ratios
  preprocessing_check_<folder>_side_views_pNN.png  pages of side views, 36 cases per page
  where <folder> is the step-2 folder name, e.g. verse19_validation_2mm_lumbar_crop.

    python scripts/03_verify_preprocessed_cases.py
    python scripts/03_verify_preprocessed_cases.py --source verse19_validation --no-pictures
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import ndimage

from lumbar.geometry import voxel_to_world_mm
from lumbar.labels import CANONICAL, TARGET_IDS
from lumbar.preprocess import TARGET_SPACING
from lumbar.sources import QC_REPORTS_DIR, downloaded_sources, preprocessed_folder

COLORS = ["#000000", "#e6194b", "#f58231", "#ffe119", "#3cb44b", "#4363d8",
          "#a9a9a9", "#911eb4", "#f032e6", "#42d4f4"]   # bg, L1-L5, T12, sacrum, L6, T13
FRAGMENT_ML = 2.0
NPZ_SUFFIX = "_image_and_label.npz"


def compare_case_with_original(npz_path, audit_row):
    """Per structure: centre distance in mm and volume ratio, preprocessed vs. original."""
    d = np.load(npz_path)
    label, affine = d["label"], d["affine"]
    voxel_ml = abs(np.linalg.det(affine[:3, :3])) / 1000.0
    orig = json.loads(audit_row["centroids_ras"]) if isinstance(audit_row.get("centroids_ras"), str) else {}
    errs, vol_ratio = {}, {}
    for cid in np.unique(label):
        if cid == 0:
            continue
        name = CANONICAL[int(cid)]
        m = label == cid
        ml = m.sum() * voxel_ml
        if name not in orig or ml < FRAGMENT_ML:
            continue
        c = voxel_to_world_mm(affine, ndimage.center_of_mass(m))[0]
        errs[name] = float(np.linalg.norm(c - np.asarray(orig[name])))
        ref_ml = audit_row.get(f"{name}_ml")
        if pd.notna(ref_ml) and ref_ml > 0:
            vol_ratio[name] = float(ml / ref_ml)
    return errs, vol_ratio


def save_side_view_pages(folder_name, items, flags_by_case, per_page=36):
    cmap = ListedColormap(COLORS)
    pages = [items[i:i + per_page] for i in range(0, len(items), per_page)]
    for p, page in enumerate(pages, 1):
        n = int(np.ceil(np.sqrt(len(page))))
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3.6 * n))
        for ax in np.ravel(axes):
            ax.axis("off")
        for ax, (case_id, npz) in zip(np.ravel(axes), page):
            d = np.load(npz)
            img, lab = d["image"].astype(np.float32), d["label"]
            target = np.isin(lab, TARGET_IDS)
            xs = np.nonzero(target.any(axis=(1, 2)) if target.any() else (lab > 0).any(axis=(1, 2)))[0]
            x = int(np.median(xs)) if xs.size else img.shape[0] // 2
            ax.imshow(img[x].T, origin="lower", cmap="gray", vmin=0.3, vmax=0.9)   # rows: S, cols: A
            ax.imshow(np.ma.masked_equal(lab[x].T, 0), origin="lower", cmap=cmap, vmin=0,
                      vmax=len(COLORS) - 1, alpha=0.45, interpolation="nearest")
            fl = flags_by_case.get(case_id, "")
            marks = [f for f in ("ORDER_VIOLATION", "TRANSITIONAL_L6", "METAL", "FRAGMENT") if f in fl]
            ax.set_title(case_id + ("\n" + " ".join(marks) if marks else ""), fontsize=8,
                         color="red" if marks else "black")
        fig.suptitle(f"{folder_name}: mid-sagittal, head up, anterior right | "
                     "L1 red, L2 orange, L3 yellow, L4 green, L5 blue, T12 grey, sacrum purple, L6 pink",
                     fontsize=10)
        fig.tight_layout()
        out = QC_REPORTS_DIR / f"preprocessing_check_{folder_name}_side_views_p{p:02d}.png"
        fig.savefig(out, dpi=70)
        plt.close(fig)
        print(f"   pictures -> {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", help="default: every downloaded source that step 2 has processed")
    ap.add_argument("--spacing", type=float, default=TARGET_SPACING, help="must match step 2")
    ap.add_argument("--crop", choices=["lumbar", "none"], default="lumbar", help="must match step 2")
    ap.add_argument("--no-pictures", action="store_true", help="only the numbers, no side-view images")
    args = ap.parse_args()

    for src in args.source or downloaded_sources():
        folder = preprocessed_folder(src, args.spacing, args.crop)
        npzs = sorted(folder.glob(f"*{NPZ_SUFFIX}"))
        if not npzs:
            print(f"[verify] {src}: nothing preprocessed in {folder}, skipped")
            continue
        audit = pd.read_csv(QC_REPORTS_DIR / f"anomalies_per_case_{src}.csv").set_index("case_id")
        rows, items = [], []
        for npz in npzs:
            case_id = npz.name[: -len(NPZ_SUFFIX)]
            items.append((case_id, npz))
            errs, vr = compare_case_with_original(npz, audit.loc[case_id].to_dict())
            rows.append({"case_id": case_id,
                         "max_centroid_err_mm": max(errs.values()) if errs else np.nan,
                         "worst_structure": max(errs, key=errs.get) if errs else "",
                         "min_volume_ratio": min(vr.values()) if vr else np.nan,
                         "max_volume_ratio": max(vr.values()) if vr else np.nan})
        df = pd.DataFrame(rows)
        df.to_csv(QC_REPORTS_DIR / f"preprocessing_check_{folder.name}.csv", index=False)
        e = df["max_centroid_err_mm"]
        print(f"[verify] {src}: {len(df)} cases | worst centre error per case: "
              f"median {e.median():.2f} mm, max {e.max():.2f} mm ({df.loc[e.idxmax(), 'case_id']}, "
              f"{df.loc[e.idxmax(), 'worst_structure']}) | volume ratio preprocessed/original: "
              f"{df['min_volume_ratio'].min():.3f} - {df['max_volume_ratio'].max():.3f}")
        if not args.no_pictures:
            save_side_view_pages(folder.name, items, audit["flags"].fillna("").to_dict())


if __name__ == "__main__":
    main()
