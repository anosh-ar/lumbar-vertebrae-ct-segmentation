"""Step 6 (Part D, + Part C check): evaluate the trained model on both test sets.

Test sets
  in-domain:      TotalSegmentator v3 official test split (totalseg_v3_test)
  out-of-domain:  VerSe 2019 validation + test (verse19_validation, verse19_test);
                  VerSe is never used for training or for choosing the checkpoint.

What it does, per case
  1. runs the same inference as the deliverable (lumbar/inference.py) on the
     ORIGINAL CT: whole scan, no ground-truth crop
  2. merges the ground-truth labels onto the CT grid (Part A loader)
  3. scores the prediction with the rules in lumbar/evaluation.py: Dice per level,
     shifted Dice, resampling ceiling, false-positive levels
  4. Part C: vertebral body centres from the prediction and from the ground truth;
     for VerSe also the annotated centroid points (which mark the body centre)
  Results are cached per case, so the script can be stopped and resumed.

Output (for the 2 mm model; a model with another voxel size writes to
outputs/evaluation_<s>mm/ and outputs/predictions_<s>mm/ instead, see --tag)
  outputs/predictions/<source>/<case>_predicted_labels.nii.gz   all classes, CT grid
  outputs/predictions/<source>/<case>_centroids.json            body centres (assignment format)
  outputs/evaluation/per_case/<source>/<case>.json              every number for that case
  outputs/evaluation/dice_per_level.csv, centroid_distances.csv  long tables of all cases
  outputs/evaluation/evaluation_report.md                       the tables for the report

    python scripts/06_evaluate_on_test_sets.py
    python scripts/06_evaluate_on_test_sets.py --source verse19_test --limit 3
    python scripts/06_evaluate_on_test_sets.py --weights weights/lumbar_vertebrae_resenc_unet_1.5mm.pt
    python scripts/06_evaluate_on_test_sets.py --report-only [--tag 1.5mm]   # rebuild a report from its cache
"""

import argparse
import json
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from lumbar.body_centroid import centroid_json, vertebral_body_centroids
from lumbar.evaluation import dice, shifted_level
from lumbar.geometry import affine_after_reorienting, resample_onto_grid, voxel_to_world_mm
from lumbar.inference import predict_ct
from lumbar.labels import CANONICAL, NAME_TO_ID, TARGET_IDS, TARGET_LEVELS, VERSE_LABELS
from lumbar.network import load_trained_network
from lumbar.preprocess import bounding_box_of_nonzero, merge_label_files_to_canonical
from lumbar.sources import PROJECT_ROOT, QC_REPORTS_DIR, WEIGHTS_FILE, load_cases

TEST_SOURCES = ["totalseg_v3_test", "verse19_validation", "verse19_test"]
GROUP = {"totalseg_v3_test": "TotalSegmentator v3 test (in-domain)",
         "verse19_validation": "VerSe19 val + test (out-of-domain)",
         "verse19_test": "VerSe19 val + test (out-of-domain)"}
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def output_folders(tag: str):
    """(evaluation folder, predictions folder). No tag: outputs/evaluation and outputs/predictions
    (the 2 mm model); tag "1.5mm": outputs/evaluation_1.5mm and outputs/predictions_1.5mm."""
    suffix = f"_{tag}" if tag else ""
    return OUTPUTS_DIR / f"evaluation{suffix}", OUTPUTS_DIR / f"predictions{suffix}"


def verse_annotated_points(case, img) -> dict:
    """VerSe centroid annotations in world RAS mm (convention verified in Part A)."""
    ctd = case.load_verse_centroids()
    if not ctd or not ctd["direction"]:
        return {}
    aff = affine_after_reorienting(img.affine, img.shape[:3], ctd["direction"])
    out = {}
    for raw, xyz in ctd["points"].items():
        name = CANONICAL.get(VERSE_LABELS.get(raw, 0))
        if name in TARGET_LEVELS:
            out[name] = voxel_to_world_mm(aff, xyz)[0].round(2).tolist()
    return out


def evaluate_case(case, source: str, net, ckpt, device, audit_row: dict, pred_dir) -> dict:
    t0 = time.time()
    img = case.load_ct()
    res = predict_ct(net, img, ckpt["patch_size"], device, ckpt["spacing_mm"], batch_size=4)
    pred = res["label_original_grid"]
    gt, _ = merge_label_files_to_canonical(case, img)
    # resampling ceiling: ground truth -> model grid (voxel size of the checkpoint) -> back to the CT grid
    gt_model = resample_onto_grid(gt, img.affine, res["model_affine"], res["label_model_grid"].shape, order=0)
    gt_back = resample_onto_grid(gt_model, res["model_affine"], img.affine, img.shape[:3], order=0)

    box = bounding_box_of_nonzero((gt > 0) | (pred > 0) | (gt_back > 0))   # Dice only where anything is
    if box is not None:
        gt, pred, gt_back = gt[box], pred[box], gt_back[box]
    voxel_ml = float(abs(np.linalg.det(img.affine[:3, :3]))) / 1000
    truncated = str(audit_row.get("truncated") or "").split()
    fragments = str(audit_row.get("fragments") or "").split()

    levels, fp_levels = {}, []
    for c in TARGET_IDS:
        name, g, p = CANONICAL[c], gt == c, pred == c
        if not g.any():
            if p.any():
                fp_levels.append(name)
            continue
        up, down = shifted_level(c, -1), shifted_level(c, +1)
        levels[name] = {
            "gt_ml": round(float(g.sum()) * voxel_ml, 2), "pred_ml": round(float(p.sum()) * voxel_ml, 2),
            "dice": dice(p, g),
            "dice_if_pred_one_level_up": dice(pred == up, g) if up else float("nan"),
            "dice_if_pred_one_level_down": dice(pred == down, g) if down else float("nan"),
            "dice_ceiling": dice(gt_back == c, g),
            "truncated": name in truncated, "fragment": name in fragments}

    pred_c = vertebral_body_centroids(res["label_model_grid"], res["model_affine"])
    gt_c = vertebral_body_centroids(gt_model, res["model_affine"])
    ctd = verse_annotated_points(case, img) if case.dataset == "verse" else {}
    centroids = {n: {"pred_body": pred_c.get(n, {}).get("body_ras_mm"), "gt_body": gt_c.get(n, {}).get("body_ras_mm"),
                     "gt_whole": gt_c.get(n, {}).get("whole_ras_mm"), "verse_point": ctd.get(n),
                     "pred_method": pred_c.get(n, {}).get("method"), "gt_method": gt_c.get(n, {}).get("method")}
                 for n in TARGET_LEVELS if n in pred_c or n in gt_c}

    out_dir = pred_dir / source
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(res["label_original_grid"], img.affine), out_dir / f"{case.case_id}_predicted_labels.nii.gz")
    (out_dir / f"{case.case_id}_centroids.json").write_text(json.dumps(centroid_json(case.case_id, pred_c), indent=2))

    present = set(np.unique(gt_model).tolist())
    return {"case_id": case.case_id, "source": source, "subject_id": case.subject_id,
            "flags": audit_row.get("flags", ""), "gt_has_L6": NAME_TO_ID["L6"] in present,
            "gt_has_T13": NAME_TO_ID["T13"] in present, "levels": levels, "false_positive_levels": fp_levels,
            "centroids": centroids, "seconds": round(time.time() - t0, 1),
            "model_spacing_mm": ckpt["spacing_mm"]}


# --------------------------------------------------------------------------- report

def distance(a, b):
    return float(np.linalg.norm(np.subtract(a, b))) if a is not None and b is not None else np.nan


def load_all_results(sources, eval_dir) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    cases = [json.loads(p.read_text()) for s in sources for p in sorted((eval_dir / "per_case" / s).glob("*.json"))]
    lv, ct = [], []
    for r in cases:
        base = {"source": r["source"], "group": GROUP[r["source"]], "case_id": r["case_id"],
                "subject_id": r["subject_id"], "gt_has_L6": r["gt_has_L6"]}
        for name, v in r["levels"].items():
            lv.append(base | {"level": name} | v)
        for name, v in r["centroids"].items():
            ct.append(base | {"level": name,
                              "pred_body_to_gt_body_mm": distance(v["pred_body"], v["gt_body"]),
                              "pred_body_to_verse_point_mm": distance(v["pred_body"], v["verse_point"]),
                              "gt_body_to_verse_point_mm": distance(v["gt_body"], v["verse_point"]),
                              "gt_whole_to_verse_point_mm": distance(v["gt_whole"], v["verse_point"]),
                              "pred_method": v["pred_method"], "gt_method": v["gt_method"]})
    return pd.DataFrame(lv), pd.DataFrame(ct), cases


def false_positive_level_sizes(cases: list[dict], pred_dir) -> pd.DataFrame:
    """Volume of every predicted level that the ground truth does not have, read from
    the saved prediction, and whether it touches the scan border (cut off)."""
    rows = []
    for r in cases:
        if not r["false_positive_levels"]:
            continue
        img = nib.load(pred_dir / r["source"] / f"{r['case_id']}_predicted_labels.nii.gz")
        lab = np.asarray(img.dataobj)
        voxel_ml = float(abs(np.linalg.det(img.affine[:3, :3]))) / 1000
        for name in r["false_positive_levels"]:
            idx = np.argwhere(lab == NAME_TO_ID[name])
            at_edge = bool((idx.min(0) == 0).any() or (idx.max(0) == np.array(lab.shape) - 1).any())
            rows.append({"group": GROUP[r["source"]], "case_id": r["case_id"], "level": name,
                         "ml": len(idx) * voxel_ml, "at_edge": at_edge})
    return pd.DataFrame(rows)


def fmt(s: pd.Series) -> str:
    s = s.dropna()
    return f"{s.mean():.3f} / {s.median():.3f} ({len(s)})" if len(s) else "-"


def build_report(lv: pd.DataFrame, ct: pd.DataFrame, cases: list[dict], pred_dir) -> str:
    scored = lv[~lv["fragment"]]
    L = ["# Part D: evaluation on the test sets", "",
         "Made by `scripts/06_evaluate_on_test_sets.py`. Scoring rules: `lumbar/src/lumbar/evaluation.py`. "
         "Dice on the original CT grid; a level is scored only where the ground truth has it "
         "(label scraps < 2 ml excluded). Cells: mean / median (number of vertebrae).", ""]

    def level_table(df, title, col="dice"):
        rows = [f"### {title}", "", "| Test set | " + " | ".join(TARGET_LEVELS) + " | all |",
                "|---|" + "---|" * (len(TARGET_LEVELS) + 1)]
        for g, d in df.groupby("group"):
            rows.append(f"| {g} | " + " | ".join(fmt(d.loc[d["level"] == n, col]) for n in TARGET_LEVELS)
                        + f" | {fmt(d[col])} |")
        return rows + [""]

    L += level_table(scored, "Dice per vertebra (strict)")
    L += level_table(scored[~scored["truncated"]], "Dice per vertebra, complete vertebrae only (not cut off by the scan edge)")
    mm = f"{cases[0].get('model_spacing_mm', 2.0):g} mm"     # results cached before this field existed are 2 mm
    L += level_table(scored, f"Resampling ceiling (ground truth -> {mm} -> back; best possible at {mm})", "dice_ceiling")
    L += level_table(scored[~scored["gt_has_L6"]], "Dice per vertebra, without the L6 patients")

    # shifts: per case, which renaming of the predicted levels fits best
    L += ["### Level counting: does renaming the predicted levels by one step fit better?", "",
          "| Test set | cases with a scored level | best: as predicted | best: one level up | best: one level down |",
          "|---|---|---|---|---|"]
    shift_rows = []
    for (g, cid), d in scored.groupby(["group", "case_id"]):
        m = {"as predicted": d["dice"].mean(), "one level up": d["dice_if_pred_one_level_up"].mean(),
             "one level down": d["dice_if_pred_one_level_down"].mean()}
        best = max(m, key=lambda k: -1 if np.isnan(m[k]) else m[k])
        shift_rows.append({"group": g, "case_id": cid, "best": best, **m})
    sh = pd.DataFrame(shift_rows)
    for g, d in sh.groupby("group"):
        vc = d["best"].value_counts()
        L.append(f"| {g} | {len(d)} | {vc.get('as predicted', 0)} | {vc.get('one level up', 0)} | {vc.get('one level down', 0)} |")
    shifted = sh[(sh["best"] != "as predicted") & (sh[["one level up", "one level down"]].max(axis=1) > sh["as predicted"] + 0.1)]
    L += ["", "Cases where a one-level renaming raises the mean Dice by more than 0.1 (the model found the "
          "vertebrae but counted them wrong): " + (", ".join(f"{r.case_id} ({r.best}: {r['as predicted']:.2f} -> "
          f"{max(r['one level up'], r['one level down']):.2f})" for _, r in shifted.iterrows()) or "none"), ""]

    # false positives
    L += ["### Levels predicted where the ground truth has none", "",
          "Size matters: a speck of a few voxels and a whole extra vertebra are different errors. "
          "A lumbar vertebra is about 60 ml; 'vertebra-sized' = 20 ml or more.", "",
          "| Test set | cases | cases with a false-positive level | of these: no lumbar level in the ground truth "
          "| false-positive levels: median size | vertebra-sized ones |", "|---|---|---|---|---|---|"]
    fp = false_positive_level_sizes(cases, pred_dir)
    cdf = pd.DataFrame([{"group": GROUP[r["source"]], "fp": bool(r["false_positive_levels"]),
                         "no_gt": not r["levels"]} for r in cases])
    for g, d in cdf.groupby("group"):
        f = fp[fp["group"] == g] if len(fp) else fp
        L.append(f"| {g} | {len(d)} | {int(d['fp'].sum())} | {int((d['fp'] & d['no_gt']).sum())} "
                 f"| {f['ml'].median():.1f} ml ({len(f)}) | {int((f['ml'] >= 20).sum())} |")
    big = fp[fp["ml"] >= 20].sort_values("ml", ascending=False) if len(fp) else fp
    L += ["", "Vertebra-sized false-positive levels: " + (", ".join(
        f"{r.case_id} {r.level} {r.ml:.0f} ml{' (touches scan edge)' if r.at_edge else ''}"
        for r in big.itertuples()) or "none"), ""]

    # L6 patients
    l6 = [r for r in cases if r["gt_has_L6"]]
    if l6:
        L += ["### Patients with a sixth lumbar vertebra (L6)", "",
              "| Case | " + " | ".join(TARGET_LEVELS) + " | best renaming |", "|---|" + "---|" * 6]
        for r in l6:
            best = sh.loc[sh["case_id"] == r["case_id"], "best"]
            L.append(f"| {r['case_id']} | " + " | ".join(f"{r['levels'][n]['dice']:.2f}" if n in r["levels"] else "-"
                                                    for n in TARGET_LEVELS) + f" | {best.iloc[0] if len(best) else '-'} |")
        L.append("")

    # worst cases
    L += ["### Lowest mean Dice per test set (for a visual check)", ""]
    per_case = scored.groupby(["group", "case_id"])["dice"].mean().reset_index()
    flags = {r["case_id"]: r["flags"] if isinstance(r["flags"], str) else "" for r in cases}
    for g, d in per_case.groupby("group"):
        worst = d.nsmallest(8, "dice")
        L.append(f"- {g}: " + "; ".join(f"{r.case_id} {r.dice:.2f} [{flags.get(r.case_id, '')}]" for r in worst.itertuples()))
    L.append("")

    # Part C
    L += ["## Part C: vertebral body centres", "",
          "Distances in mm: median / 90th percentile (number of vertebrae). "
          "'VerSe point' = the annotated centroid in VerSe, which marks the vertebral body centre (Part A).", "",
          "| Test set | predicted body vs ground-truth body | predicted body vs VerSe point "
          "| ground-truth body vs VerSe point | ground-truth whole vertebra vs VerSe point |", "|---|---|---|---|---|"]

    def q(s):
        s = s.dropna()
        return f"{s.median():.1f} / {s.quantile(0.9):.1f} ({len(s)})" if len(s) else "-"
    for g, d in ct.groupby("group"):
        L.append(f"| {g} | {q(d['pred_body_to_gt_body_mm'])} | {q(d['pred_body_to_verse_point_mm'])} | "
                 f"{q(d['gt_body_to_verse_point_mm'])} | {q(d['gt_whole_to_verse_point_mm'])} |")
    real = ct[ct["gt_method"].notna() & ct["pred_method"].notna()]
    fb = real["pred_method"].eq("whole_vertebra_fallback").sum()
    L += ["", f"Of {len(real)} predicted vertebrae that also exist in the ground truth, {fb} were too small to "
          "separate the body (mostly cut off by the scan edge); their whole-vertebra centre is used.", ""]

    t = [r["seconds"] for r in cases]
    L += [f"Inference + scoring time per case on this machine: median {np.median(t):.0f} s, max {np.max(t):.0f} s.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", default=TEST_SOURCES)
    ap.add_argument("--weights", default=WEIGHTS_FILE)
    ap.add_argument("--limit", type=int, help="only the first N cases per source, for quick tests")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--report-only", action="store_true", help="only rebuild the report from cached results")
    ap.add_argument("--tag", help="results folder suffix, e.g. 1.5mm -> outputs/evaluation_1.5mm; default: "
                                  "none for a 2 mm model, '<voxel size>mm' otherwise (with --report-only: none)")
    args = ap.parse_args()

    if not args.report_only:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else ("cpu" if args.device == "auto" else args.device))
        net, ckpt = load_trained_network(args.weights, device)
        if args.tag is None:   # never mix models: the voxel size of the weights picks the folders
            args.tag = "" if ckpt["spacing_mm"] == 2.0 else f"{ckpt['spacing_mm']:g}mm"
    eval_dir, pred_dir = output_folders(args.tag or "")
    if not args.report_only:
        print(f"[evaluate] weights {args.weights} from training step {ckpt['step']} "
              f"(validation Dice {ckpt['best_val_dice']:.3f}, {ckpt['spacing_mm']:g} mm voxels), {device} "
              f"-> {eval_dir.name}/, {pred_dir.name}/")
        for src in args.source:
            audit = pd.read_csv(QC_REPORTS_DIR / f"anomalies_per_case_{src}.csv").set_index("case_id")
            cache = eval_dir / "per_case" / src
            cache.mkdir(parents=True, exist_ok=True)
            cases = load_cases(src)[:args.limit]
            print(f"[evaluate] {src}: {len(cases)} cases", flush=True)
            for i, case in enumerate(cases, 1):
                out = cache / f"{case.case_id}.json"
                if out.exists():
                    continue
                row = audit.loc[case.case_id].to_dict() if case.case_id in audit.index else {}
                r = evaluate_case(case, src, net, ckpt, device, row, pred_dir)
                out.write_text(json.dumps(r, indent=1))
                d = [v["dice"] for v in r["levels"].values()]
                print(f"   {i}/{len(cases)} {case.case_id}: mean Dice {np.mean(d) if d else float('nan'):.3f} "
                      f"over {len(d)} levels, false-positive levels {r['false_positive_levels'] or '-'} "
                      f"({r['seconds']:.0f} s)", flush=True)

    lv, ct, cases = load_all_results(args.source, eval_dir)
    lv.to_csv(eval_dir / "dice_per_level.csv", index=False)
    ct.to_csv(eval_dir / "centroid_distances.csv", index=False)
    (eval_dir / "evaluation_report.md").write_text(build_report(lv, ct, cases, pred_dir), encoding="utf-8")
    print(f"[evaluate] report -> {eval_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
