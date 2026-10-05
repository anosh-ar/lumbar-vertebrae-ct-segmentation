"""Step 5 (Parts B + C): segment L1-L5 and find the vertebral body centres in new CTs.

This is the inference script of the deliverable (run_inference.sh calls it).
It needs only the CT files and the trained weights; no labels, no GPU.

What it does, per CT
  1. canonical space as in training: RAS, the voxel size stored in the weights file
     (2 or 1.5 mm), HU window -> 0..1 (whole scan, no crop)
  2. sliding-window segmentation with the trained network, largest piece per class
  3. vertebral body centre per level L1-L5 (lumbar/body_centroid.py), world mm, RAS
  4. labels resampled back onto the CT's own grid

Input
  a folder with CT files (*.nii.gz or *.nii, any orientation), searched recursively.
  Case id = file name without extension and without a trailing "_ct"; a file
  named just "ct.nii.gz" (TotalSegmentator layout) takes its folder's name.

Output (one pair per CT)
  <case>_lumbar_labels.nii.gz   same grid as the input CT; 0 background, 1-5 = L1-L5
  <case>_centroids.json         {"case_id", "coordinate_system": "RAS", "unit": "mm", "centroids": {...}}

    python scripts/05_predict_lumbar_vertebrae.py <input_dir> <output_dir>
    python scripts/05_predict_lumbar_vertebrae.py <input_dir> <output_dir> --device cpu
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from lumbar.body_centroid import centroid_json, vertebral_body_centroids
from lumbar.inference import predict_ct
from lumbar.labels import TARGET_IDS
from lumbar.network import load_trained_network
from lumbar.sources import WEIGHTS_FILE


def case_id_from_path(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    if name == "ct":
        return path.parent.name
    return name[:-3] if name.endswith("_ct") else name


def save_label_like(label: np.ndarray, ref: nib.Nifti1Image, path: Path):
    """uint8 label volume with the reference CT's grid (both qform and sform set)."""
    out = nib.Nifti1Image(label.astype(np.uint8), ref.affine)
    out.set_qform(ref.affine, code=1)
    out.set_sform(ref.affine, code=1)
    nib.save(out, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--weights", type=Path, default=WEIGHTS_FILE)
    ap.add_argument("--device", default="auto", help="auto (GPU if available), cpu or cuda")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    files = sorted(p for p in args.input_dir.rglob("*") if p.name.endswith((".nii.gz", ".nii")))
    if not files:
        raise SystemExit(f"no .nii/.nii.gz files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    net, ckpt = load_trained_network(args.weights, device)
    print(f"[predict] {len(files)} CTs on {device}; weights {args.weights.name} (training step {ckpt['step']})")

    for path in files:
        t0 = time.time()
        cid = case_id_from_path(path)
        img = nib.load(path)
        res = predict_ct(net, img, ckpt["patch_size"], device, ckpt["spacing_mm"])
        lumbar_only = np.where(np.isin(res["label_original_grid"], TARGET_IDS), res["label_original_grid"], 0)
        save_label_like(lumbar_only, img, args.output_dir / f"{cid}_lumbar_labels.nii.gz")
        cents = vertebral_body_centroids(res["label_model_grid"], res["model_affine"])
        (args.output_dir / f"{cid}_centroids.json").write_text(json.dumps(centroid_json(cid, cents), indent=2))
        print(f"   {cid}: levels {' '.join(cents) or 'none'}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
