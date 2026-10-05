"""Step 2 (Part A): turn every case into small, uniform arrays for training.

What it does, per case
  1. merge the label files into one mask with our label ids (lumbar/labels.py)
  2. crop to the labelled region T12..sacrum plus 30 mm
  3. rotate to RAS and resample to cubic voxels of --spacing mm, default 2 (CT: smoothed where it is
     downsampled, then linear interpolation; mask: nearest neighbour)
  4. clip the CT to -1000..1500 HU and scale it to 0..1

Output (data/preprocessed/<source>_<spacing>mm_lumbar_crop/, e.g. ..._2mm_..., ..._1.5mm_...)
  <case>_image_and_label.npz   CT (float16, 0..1), mask (uint8), affine (voxel -> RAS mm)
  <case>_provenance.json       where it came from: original header, crop box, fixes applied
  preprocessing_status.csv     one row per case: written / already_done / skipped_no_spine / error

The crop is placed using the ground-truth labels, so this output is for
training and checks only. Evaluation must run inference on the original files.
Cases with none of T12, L1-L5, L6, T13 or sacrum are skipped.

Sources: all downloaded ones by default. Safe to re-run: finished cases are
kept unless --overwrite. About 1.5 s per case with 3 workers.

    python scripts/02_preprocess_cases_for_training.py
    python scripts/02_preprocess_cases_for_training.py --source totalseg_v3_train --workers 4
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from lumbar.preprocess import TARGET_SPACING, preprocess_case_to_canonical
from lumbar.sources import downloaded_sources, load_cases, preprocessed_folder

STATUS_FILE = "preprocessing_status.csv"


def preprocess_and_save_case(case, out_dir, spacing, crop, overwrite):
    npz = out_dir / f"{case.case_id}_image_and_label.npz"
    provenance = out_dir / f"{case.case_id}_provenance.json"
    row = {"case_id": case.case_id, "dataset": case.dataset, "split": case.split,
           "subject_id": case.subject_id}
    if npz.exists() and provenance.exists() and not overwrite:
        meta = json.loads(provenance.read_text())
        return row | {"status": "already_done", "shape": meta["shape"],
                      "labels": " ".join(meta["voxels_per_label"])}
    try:
        res = preprocess_case_to_canonical(case, spacing=spacing, crop=crop)
    except Exception as e:  # noqa: BLE001
        return row | {"status": "error", "error": repr(e)}
    if res is None:
        return row | {"status": "skipped_no_spine"}
    np.savez_compressed(npz, image=res["image"].astype(np.float16), label=res["label"], affine=res["affine"])
    provenance.write_text(json.dumps(res["meta"], indent=1, default=str))
    return row | {"status": "written", "shape": res["meta"]["shape"],
                  "labels": " ".join(res["meta"]["voxels_per_label"]),
                  "fixes": ";".join(res["meta"]["intensity_fixes"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", help="default: every downloaded source")
    ap.add_argument("--spacing", type=float, default=TARGET_SPACING, help="voxel size in mm (default 2)")
    ap.add_argument("--crop", choices=["lumbar", "none"], default="lumbar",
                    help="'none' keeps the whole scan (written to a separate *_full_image folder)")
    ap.add_argument("--workers", type=int, default=3, help="parallel processes; each holds one full CT in memory")
    ap.add_argument("--limit", type=int, help="only the first N cases per source, for quick tests")
    ap.add_argument("--overwrite", action="store_true", help="redo cases that are already done")
    args = ap.parse_args()

    for src in args.source or downloaded_sources():
        cases = load_cases(src)[:args.limit]
        out_dir = preprocessed_folder(src, args.spacing, args.crop)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[preprocess] {src}: {len(cases)} cases -> {out_dir}")
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(preprocess_and_save_case, c, out_dir, args.spacing, args.crop, args.overwrite)
                    for c in cases]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 10 == 0 or i == len(cases):
                    print(f"   {i}/{len(cases)}  {time.time() - t0:.0f}s")
        df = pd.DataFrame(rows).sort_values("case_id")
        df.to_csv(out_dir / STATUS_FILE, index=False)
        print("   status:", df["status"].value_counts().to_dict())
        if (df["status"] == "error").any():
            print(df.loc[df["status"] == "error", ["case_id", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
