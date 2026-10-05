"""Step 8: compare two evaluated models side by side (e.g. 2 mm vs 1.5 mm voxels).

What it does
  Reads the results that step 6 wrote for each model (outputs/evaluation*/ and
  outputs/predictions*/) and puts the key numbers next to each other:
  Dice per vertebra (mean / median; all vertebrae and only complete ones), the
  resampling ceiling, counting errors, false-positive levels, body-centre distances,
  and, if outputs/cpu_inference_seconds_per_scan.csv exists, the CPU time per scan.

Output
  outputs/evaluation_comparison_<a>_vs_<b>.md

    python scripts/08_compare_two_evaluations.py                  # 2mm (no tag) vs 1.5mm
    python scripts/08_compare_two_evaluations.py --tags "" 1.5mm --names 2mm 1.5mm
"""

import argparse
import json

import nibabel as nib
import numpy as np
import pandas as pd

from lumbar.labels import NAME_TO_ID, TARGET_LEVELS
from lumbar.sources import PROJECT_ROOT

OUTPUTS = PROJECT_ROOT / "outputs"
GROUPS = {"TotalSegmentator v3 test (in-domain)": "TotalSeg test", "VerSe19 val + test (out-of-domain)": "VerSe19"}
VERTEBRA_SIZED_ML = 20.0


def folders(tag: str):
    suffix = f"_{tag}" if tag else ""
    return OUTPUTS / f"evaluation{suffix}", OUTPUTS / f"predictions{suffix}"


def summarise_model(tag: str) -> dict:
    """Every number of the comparison table for one model, keyed by (row, test set)."""
    eval_dir, pred_dir = folders(tag)
    lv = pd.read_csv(eval_dir / "dice_per_level.csv")
    ct = pd.read_csv(eval_dir / "centroid_distances.csv")
    lv = lv[~lv["fragment"]]
    out = {}
    for group, short in GROUPS.items():
        d = lv[lv["group"] == group]
        complete = d[~d["truncated"]]
        for name in TARGET_LEVELS + ("all",):
            a = d if name == "all" else d[d["level"] == name]
            c = complete if name == "all" else complete[complete["level"] == name]
            out[(f"Dice {name}: all, mean / median", short)] = f"{a['dice'].mean():.3f} / {a['dice'].median():.3f}"
            out[(f"Dice {name}: not cut off, mean / median", short)] = f"{c['dice'].mean():.3f} / {c['dice'].median():.3f}"
        out[("Resampling ceiling, mean", short)] = f"{d['dice_ceiling'].mean():.3f}"
        out[("Vertebrae scored (cases)", short)] = f"{len(d)} ({d['case_id'].nunique()})"

        # counting errors: cases where renaming every predicted level by one step adds > 0.1 mean Dice
        per_case = d.groupby("case_id")[["dice", "dice_if_pred_one_level_up", "dice_if_pred_one_level_down"]].mean()
        gain = per_case[["dice_if_pred_one_level_up", "dice_if_pred_one_level_down"]].max(axis=1) - per_case["dice"]
        miscounted = sorted(per_case.index[gain > 0.1])
        out[("Cases counted one level off", short)] = f"{len(miscounted)}: {', '.join(miscounted) or '-'}"

        # body centres
        c = ct[ct["group"] == group]
        q = c["pred_body_to_gt_body_mm"].dropna()
        out[("Body centre: predicted vs ground-truth body, median / 90th pct (mm)", short)] = \
            f"{q.median():.1f} / {q.quantile(0.9):.1f}"
        v = c["pred_body_to_verse_point_mm"].dropna()
        out[("Body centre: predicted vs VerSe annotated point, median / 90th pct (mm)", short)] = \
            f"{v.median():.1f} / {v.quantile(0.9):.1f}" if len(v) else "-"

    # false-positive levels (predicted, absent from the ground truth), with their size
    rows = []
    for p in sorted((eval_dir / "per_case").glob("*/*.json")):
        r = json.loads(p.read_text())
        if not r["false_positive_levels"]:
            rows.append({"group": r["source"], "case": r["case_id"], "ml": None})
            continue
        img = nib.load(pred_dir / r["source"] / f"{r['case_id']}_predicted_labels.nii.gz")
        lab, voxel_ml = np.asarray(img.dataobj), float(abs(np.linalg.det(img.affine[:3, :3]))) / 1000
        for n in r["false_positive_levels"]:
            rows.append({"group": r["source"], "case": r["case_id"], "ml": float((lab == NAME_TO_ID[n]).sum()) * voxel_ml})
    fp = pd.DataFrame(rows)
    fp["short"] = np.where(fp["group"].str.startswith("totalseg"), "TotalSeg test", "VerSe19")
    for short, d in fp.groupby("short"):
        f = d.dropna(subset=["ml"])
        out[("False-positive levels: cases with any / levels / median size", short)] = \
            f"{f['case'].nunique()} / {len(f)} / {f['ml'].median():.1f} ml"
        out[(f"False-positive levels: vertebra-sized (>= {VERTEBRA_SIZED_ML:g} ml)", short)] = f"{int((f['ml'] >= VERTEBRA_SIZED_ML).sum())}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs=2, default=["", "1.5mm"], help="evaluation folder tags of the two models")
    ap.add_argument("--names", nargs=2, default=["2mm", "1.5mm"], help="column names")
    args = ap.parse_args()

    a, b = (summarise_model(t) for t in args.tags)
    rows = list(dict.fromkeys(k[0] for k in a))
    L = [f"# Model comparison: {args.names[0]} vs {args.names[1]}", "",
         "Made by `scripts/08_compare_two_evaluations.py` from the step-6 results. Dice on the original CT grid.", "",
         f"| | TotalSeg test {args.names[0]} | TotalSeg test {args.names[1]} | VerSe19 {args.names[0]} | VerSe19 {args.names[1]} |",
         "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r} | {a.get((r, 'TotalSeg test'), '-')} | {b.get((r, 'TotalSeg test'), '-')} "
                 f"| {a.get((r, 'VerSe19'), '-')} | {b.get((r, 'VerSe19'), '-')} |")
    timing = OUTPUTS / "cpu_inference_seconds_per_scan.csv"
    if timing.exists():
        t = pd.read_csv(timing)
        L += ["", "CPU-only inference time per scan (same scans, same machine, seconds):", "",
              "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
        L += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    out = OUTPUTS / f"evaluation_comparison_{args.names[0]}_vs_{args.names[1]}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[compare] -> {out}")


if __name__ == "__main__":
    main()
