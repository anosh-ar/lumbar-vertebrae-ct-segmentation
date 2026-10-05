# Handoff: Lumbar Spine L1–L5 take-home assignment

State as of 2026-10-01 (Parts A–D done, report written). Written for the next AI assistant working on this project.
Everything below was checked against the files on disk when this was written.

---

## 0. Read first

- **Wait for the user's go-ahead before writing or changing code.** The user's latest standing
  instruction was: *"Don't continue developing the code until I tell you."* Moving files, running
  existing scripts to answer a question, and explaining are fine; new code or edits are not.
- **Explain in plain, easy language.** The user regularly asks "explain X in easy language" and
  "what does this part do". They write quickly (typos are normal) and value honest answers,
  including "this is not implemented yet".
- **The project moved to a stronger workstation (2026-09-30).** The code is in a private GitHub
  repository; data, outputs and weights are not (see `.gitignore`). On the workstation the user
  downloads the zips by hand into `data/archives/` and runs everything from step 00 (`README.md`).
  The old Windows laptop (16 GB RAM, no GPU) is no longer the main machine.
- **Git commits are authored by the user only** (their own git identity); never add Claude
  co-author or attribution lines.
- Do not extract the 35 GB TotalSegmentator v3 zip; the code reads it in place.

---

## 1. The assignment (docs/Take-Home_Assignment_Lumbar_Spine_L1-L5.pdf)

Effort ~1 working day; deadline 5 days after receipt (PDF saved 2026-09-29; confirm the exact
date with the user). Main deliverable: **two-page PDF report**. Graded on sound engineering and
clear reasoning, not top score. A 30-minute code review follows: the user must be able to explain
every design decision, so keep code legible and choices explicit.

- **Data:** TotalSegmentator (use the *official* train/test split) and VerSe 2019/2020.
  Targets: L1, L2, L3, L4, L5.
- **Part A, preprocessing:** one loader for both datasets; canonical orientation, common voxel
  spacing, intensity normalisation. **Report every header anomaly: how detected, how handled.**
- **Part B, segmentation:** train a 3D model on TotalSegmentator; L1–L5 as separate classes.
  Any framework. **Pretrained spine models are forbidden** (TotalSegmentator, VerSe models).
- **Part C, postprocessing:** per vertebra, the centroid of the **vertebral body only** (no
  pedicles, laminae, processes). No dataset has a body label. One JSON per case:
  `{"case_id", "coordinate_system": "RAS"|"LPS", "unit": "mm", "centroids": {"L2": [x,y,z], ...}}`
  in world mm. Optional: validate against VerSe centroid annotations and explain what they describe.
- **Part D, cross-dataset evaluation:** evaluate on the TotalSegmentator test split (in-domain) and
  VerSe (out-of-domain). Harmonise labels, document the mapping incl. transitional vertebrae (L6).
  At minimum Dice per vertebra L1–L5 on both. Explain differences; "did you notice anything odd?"
- **Deliverables:** (1) 2-page PDF report covering A–D + "what next with more resources, and what
  you would deliberately *not* do"; tables and figures count toward the 2 pages. (2) one zip with
  code + model weights + everything for inference, **no training data**. (3) an inference script,
  single command on a fresh machine, e.g. `./run_inference.sh <input_dir> <output_dir>`; document
  OS, GPU yes/no, Docker or not.
- **Rules:** AI assistants allowed. If short on time, a working end-to-end pipeline with honest
  reporting beats one polished part. GDPR and dataset licences are the user's responsibility.
- PDF hardware tips: crop to the lumbar region and resample to 1.5–2 mm; patches + mixed
  precision; preprocess on CPU locally and upload only prepared data; CPU inference is fine.

---

## 2. Environment

- Setup on any machine: `README.md` (venv at the repo root, `pip install -e lumbar`, Python 3.10+).
  Versions last used on the laptop: Python 3.12.5, numpy 2.5.3, scipy 1.18.1, nibabel 5.4.2,
  pandas 3.0.6, matplotlib 3.11.2. **torch/MONAI are not installed yet** (needed for Part B).
- Data root defaults to `<repo>/data`; override with env var `LUMBAR_DATA`.
- **Workstation (main machine since 2026-10-01):** Windows 11, Intel i9-10980XE (18 cores / 36
  threads), 128 GB RAM, **NVIDIA RTX 3090 24 GB** (driver 555.85), D: 363 GB free (C: only ~44 GB).
  Repo at `D:\Users\za76buso\temp\lumbar-spine-segmentation` (no spaces in the path).
  No plain Python: conda only. Environment **`EMGNet`** (Python 3.12.14, `pip install -e lumbar`;
  same package versions as above). conda is *not* on PATH in the assistant's shell: call
  `C:\Users\za76buso\AppData\Local\miniconda3\envs\EMGNet\python.exe` directly
  (conda.exe: `C:\Users\za76buso\AppData\Local\miniconda3\Scripts\conda.exe`).
  The user said to ask before installing anything into an existing environment.
- Git: repository initialised 2026-09-30 on branch `main`; `.gitignore` excludes `data/`,
  `outputs/`, `.venv/`, weights (`*.pt`, `*.pth`, `*.ckpt`), zips and NIfTI files.

### Tooling pitfalls already hit
- One long bash heredoc (Python source with many quotes) failed to parse; writing that file with
  the file-write tool worked. Short heredocs are fine.
- pandas: `df.flags` is a built-in DataFrame attribute. Always use `df["flags"]`.
- Scripts use `ProcessPoolExecutor`; keep the `if __name__ == "__main__":` guard (Windows spawn).

---

## 3. Folder layout

```
<repo>/
├── README.md, HANDOFF.md, CLAUDE.md, .gitignore, .gitattributes
├── docs/Take-Home_Assignment_Lumbar_Spine_L1-L5.pdf
├── lumbar/                   THE CODE DELIVERABLE (this folder gets zipped)
│   ├── pyproject.toml
│   ├── src/lumbar/{labels,cases,sources,geometry,preprocess,audit}.py
│   └── scripts/{00_extract_verse_archives, 01_find_header_and_label_anomalies,
│                02_preprocess_cases_for_training, 03_verify_preprocessed_cases}.py
├── data/                     NOT in git; created on each machine
│   ├── archives/             zips downloaded by hand: Totalsegmentator_dataset_v300.zip (35 GB,
│   │                         read in place), dataset-verse19validation.zip, dataset-verse19test.zip
│   ├── raw/verse19/{validation,test}/{rawdata,derivatives}/sub-*/...   (step 00)
│   └── preprocessed/<source>_2mm_lumbar_crop/                          (step 02)
└── outputs/qc/               NOT in git: anomalies_per_case_<source>.csv, anomaly_report.md,
                              preprocessing_check_<folder>.csv + _side_views_pNN.png
```

**Renaming done 2026-09-30 at the user's request** ("names must say what they do; 3-5 words is
fine"). The `totalseg_small` source (v2.0.1 subset) was removed from the code and its files were
deleted: the full v3 dataset replaces it. Every script defaults to all downloaded full sources:
`totalseg_v3_train`, `totalseg_v3_test`, `verse19_validation`, `verse19_test`
(`verse19_training` is defined but not downloaded).

---

## 4. Data facts

### TotalSegmentator
- **v3.0.0** (`_v300.zip`, Zenodo record 22688904): 1939 CTs, 117 structures, one binary mask per
  structure in `sXXXX/segmentations/`. Zip has a top folder `Totalsegmentator_dataset_v300/`.
  `meta.csv` (`;`-separated, utf-8-sig) has a `split` column: **1830 train / 109 test, no val**.
  v3 added 291 paediatric CTs and "fixed the vertebrae label mixups" of v2. **Use v3 for the
  official split, training and in-domain test.**
- Lumbar coverage in v3, **measured by the full audit (2026-10-01)**:
  train 1830 = 750 all L1–L5 / 387 some / 693 none;
  test 109 = **57 all L1–L5 / 25 some / 27 none** (in-domain test n = 82 cases with any lumbar level).
- **Small subset** (`_small_v201.zip`): 102 subjects, **v2.0.1 labels** (pre-fix), split
  98 train / 4 val / **0 test**. A dev/debug set only; it cannot provide the official test split.
- Spine mask files: `vertebrae_C1..C7, T1..T12, L1..L5, vertebrae_S1, sacrum`. S1 and sacrum
  are disjoint, but S1 overlaps L5 at the junction.
- All small-subset CTs: RAS, 1.5 mm isotropic, int16, sform only (qform_code 0, sform_code 2).

### VerSe 2019 (only VerSe19 is used; VerSe20 not downloaded, by decision: VerSe is test-only)
- Layout: `rawdata/<sub>/<case>_ct.nii.gz` (+ `_ct.json` scanner sidecar) and
  `derivatives/<sub>/<case>_seg-vert_msk.nii.gz`, `_seg-subreg_ctd.json`, `_seg-vert_snp.png`.
- `<case>` can include a split id: `sub-verse400_split-verse090` (one subject, several scans).
- Labels: 1–7 C1–C7, 8–19 T1–T12, 20–24 L1–L5, **25 L6**, 26 sacrum, 27 coccyx, **28 T13**.
- `validation`: 40 scans. `test`: 40 scans, plus one unusual extra file
  `derivatives/sub-verse012/sub-verse012_seg-subreg_msk.nii.gz` (a subregion mask; not inspected;
  possibly relevant to Part C; ignored by the loader).
- The zips contain `__MACOSX/` junk; the loader and extractor skip it.
- **Centroid file convention, verified empirically** (audit, VerSe19 validation): the first entry
  is `{"direction": ["P","I","R"]}` (nibabel axcodes); the X/Y/Z values are **voxel indices of the
  image reoriented to that direction** (median distance to the mask centre of mass 12.1 mm on
  anisotropic scans vs 39.0 mm for the "mm" reading). The annotated points sit a median
  **+11.7 mm anterior** of the whole-vertebra centre of mass, i.e. they mark the **vertebral body**
  centre. That is the key Part C validation argument.

---

## 5. Code map (lumbar/src/lumbar)

- **labels.py**: canonical label ids used everywhere: 0 bg, 1–5 L1–L5, 6 T12, 7 sacrum
  (6 and 7 are auxiliary "anchor" classes to help level counting), 8 L6, 9 T13 (VerSe only, kept
  so transitional anatomy never silently becomes background). `NUM_TRAIN_CLASSES = 8` (ids 0–7).
  `TOTALSEG_MASKS` maps file stem → id, in **merge order**: T12, sacrum, S1 first, then L1–L5, so
  target levels win where masks overlap. `VERSE_LABELS` maps VerSe ids → canonical.
- **cases.py**: `FolderOrZipReader` reads files from a folder *or* a zip (NIfTI from zip: bytes →
  gunzip → `Nifti1Image.from_bytes`); `open_folder_or_zip` caches one per process. `Case` is a
  small picklable record of *paths*, not data; loads lazily (`load_ct`, `load_label_file`,
  `load_verse_centroids`). `LabelFile` = one label file + `value_to_canonical_id`
  (TotalSegmentator: 8 files `{1: id}`; VerSe: 1 file with the full mapping).
  `find_totalsegmentator_cases(root, splits)` reads meta.csv and auto-detects a top-folder prefix;
  `find_verse_cases(root, split)` finds `*_ct.nii.gz` under rawdata and pairs derivatives.
- **sources.py**: named sources → (path, finder): `totalseg_v3_train`, `totalseg_v3_test` (both
  from the v3 zip), `verse19_training` (not downloaded), `verse19_validation`, `verse19_test`.
  `load_cases(name)`, `downloaded_sources()` (the default for every script),
  `preprocessed_folder(source, spacing, crop)` → `data/preprocessed/<source>_2mm_lumbar_crop`
  (settings in the folder name so crops and full images never mix). Path constants:
  `ARCHIVES_DIR`, `EXTRACTED_DIR`, `PREPROCESSED_DIR`, `QC_REPORTS_DIR`.
- **geometry.py**: `read_header_summary`, `has_no_value_scaling`, `max_affine_difference`,
  `voxel_to_world_mm`, `affine_after_reorienting` (for VerSe centroids), `resample_onto_grid`
  (scipy `affine_transform` through inv(src)·dst: handles permutation, flips, spacing and shear in
  one step), `ras_grid_covering_volume` (axis-aligned RAS grid covering a (sub)volume),
  `antialias_sigma_per_axis` (σ = (factor−1)/2 per source axis, 0 where not downsampled).
  `AFFINE_TOL = 1e-3`.
- **preprocess.py** (Part A): `TARGET_SPACING = 2.0`, `HU_WINDOW = (-1000, 1500)`,
  `ROI_MARGIN_MM = 30`. `load_ct_as_hu` (float32, NIfTI scaling applied; fixes: drop 4th dim,
  non-finite → −1024, `hu_offset_is_missing` → −1024 offset). `normalise_hu_to_0_1` (clip to
  window, scale to [0,1]; deliberately fixed, not per-case z-score). `bounding_box_of_nonzero`.
  `merge_label_files_to_canonical(case, ref_img)` (all per-voxel work inside each mask's bounding
  box: 47 s → 5 s per case; rounds float labels, NN-resamples labels not on the image grid, counts
  overlaps named like `"sacrum&vertebrae_L5"`). `lumbar_region_bounds` (bbox of ids 1–9 + margin,
  None if none present). `preprocess_case_to_canonical(case, spacing, crop="lumbar"|"none")`:
  load → merge → crop → Gaussian anti-alias → resample (image trilinear `order=1`, labels nearest
  `order=0`) → normalise; returns image, label, affine, meta (provenance).
- **audit.py** (Part A): `audit_case_for_anomalies(case)` → one dict row with measurements and
  space-separated anomaly codes in `row["flags"]`. `ANOMALIES` = {code: `Anomaly(problem,
  how_found, what_we_do)`}, all in plain language; the report table is generated from it.
  "Plan (Part D): ..." in `what_we_do` marks handling not implemented yet. Codes: LOAD_ERROR,
  NO_XFORM, QS_MISMATCH, BAD_PIXDIM, SHEAR, NONAXIAL, ANISO, IMG_4D, IMG_SCALED, HU_OFFSET,
  NONFINITE, PAD_OUTSIDE_FOV, METAL (inside the lumbar ROI), LABEL_GRID_MISMATCH,
  LABEL_AFFINE_EPS, LABEL_FLOAT, LABEL_NONINT, LABEL_SCALED, MASK_OVERLAP, NO_LUMBAR, LEVEL_GAP,
  TRUNCATED, MULTI_CC, FRAGMENT (< 2 ml), VOLUME_OUTLIER, ORDER_VIOLATION (fragments ignored),
  TRANSITIONAL_L6, TRANSITIONAL_T13, SPLIT_SCAN, CTD_LABEL_MISMATCH. It also compares VerSe ctd
  points with mask centroids (see §4).

### Scripts (run from `lumbar/` with `python scripts/<name>.py`, venv active)
With no `--source`, each script processes every downloaded source. `--limit N` for quick tests.
- **00_extract_verse_archives.py**: unzips `dataset-verse19<split>.zip` into
  `data/raw/verse19/<split>/` (skips `__MACOSX`). v3 is never extracted. Idempotent.
- **01_find_header_and_label_anomalies.py** `[--source ...] [--workers 3] [--limit N]`: writes
  `outputs/qc/anomalies_per_case_<source>.csv` and rebuilds `outputs/qc/anomaly_report.md` from
  all such CSVs. Report columns: Problem | Code | counts per dataset | How we find it | What we do.
- **02_preprocess_cases_for_training.py** `[--source ...] [--spacing 2] [--crop lumbar|none]
  [--workers 3] [--limit N] [--overwrite]`: writes into `data/preprocessed/<source>_2mm_lumbar_crop/`
  `<case>_image_and_label.npz` (image float16 [0,1], label uint8, affine 4x4),
  `<case>_provenance.json`, `preprocessing_status.csv` (written / already_done / skipped_no_spine /
  error). Resumable. **The crop uses ground-truth labels: training/checks only; evaluation must run
  the inference path on the original files.**
- **03_verify_preprocessed_cases.py** `[--source ...] [--spacing] [--crop] [--no-pictures]`:
  compares every structure's world centroid and volume after preprocessing with step 1's CSV
  (original grid), writes `outputs/qc/preprocessing_check_<folder>.csv` and
  `..._side_views_pNN.png` (mid-sagittal, head up, anterior right). Needs step 1's
  `centroids_ras` and `<level>_ml` columns.

Runtime on the old laptop, 3 workers: ≈1.5 s per case for step 1 and for step 2.
Runtime on the workstation, `--workers 12`, all 2019 cases: step 00 14 s, step 01 3 min,
step 02 3.3 min, step 03 4 min. Peak memory per worker 2.4 GB (so 12 workers is far from the limit).
Step 00 has no `--limit`. Step 01/02 print progress every 10 cases.

---

## 6. Results so far

### Full run on the workstation (2026-10-01): steps 00–03 on all four sources, no errors

Outputs: `outputs/qc/anomaly_report.md`, `anomalies_per_case_<source>.csv`,
`preprocessing_check_<folder>.csv` + side views; arrays in `data/preprocessed/<source>_2mm_lumbar_crop/`.

**Step 01 (anomalies), counts in cases, train / test / VerSe19 val / VerSe19 test:**
- v3 headers are uniform: all RAS, 1.5 mm isotropic, int16, sform-only. No header problems at all.
  Every problem in v3 is in the labels: MASK_OVERLAP 760 / 57, TRUNCATED 402 / 29,
  MULTI_CC 156 / 10, FRAGMENT 135 / 11, METAL 118 / 6, PAD_OUTSIDE_FOV 92 / 12,
  VOLUME_OUTLIER 31 / 1, LEVEL_GAP 1 (s1048, train).
- VerSe19 test: PIR 17 / LAS 15 / PSR 8, NONAXIAL 25, ANISO 2 (sub-verse040,
  sub-verse414_split-verse241), qform-only 37, float32 CT 39, LABEL_FLOAT 1 (sub-verse012),
  METAL 2 (sub-verse040, sub-verse144), NO_LUMBAR 4, TRUNCATED 5, SPLIT_SCAN 6.
  VerSe19 val numbers match §6 below (NONAXIAL 23, ANISO 2, METAL 2: sub-verse093, sub-verse264).
- **L6 cases: sub-verse023 (val), sub-verse083 and sub-verse131 (test).** No T13 anywhere.
  No ORDER_VIOLATION, no LOAD_ERROR, no NO_XFORM/BAD_PIXDIM/SHEAR/IMG_SCALED/HU_OFFSET anywhere.
- VerSe lumbar coverage: test 31 all / 5 some / 4 none; val 26 / 5 / 9.
- **VerSe ctd = vertebral body centre, confirmed on both splits:** voxel-index reading 12.1 mm
  (test) / 12.2 mm (val) from the mask centre vs 26.3 / 17.9 mm for the mm reading; the point sits
  a median **+11.75 mm (test) / +11.71 mm (val) anterior** of the whole-vertebra centre
  (IQR test 11.5–13.1, val 10.9–12.5). The near-identical medians are real, not a bug.

**Step 02 (preprocessing, 2 mm, lumbar crop):** written train 1257 / test 87 / val 32 / test 36;
skipped (no spine label) 573 / 22 / 8 / 4; 0 errors; 1.9 GB total (train 1.7 GB).
**Of the written crops, 120 train, 5 test and 1 VerSe val are NO_LUMBAR** (T12/sacrum only;
see issue 1). Several of those are crops around a 0.1–0.2 ml T12 scrap that look like noise
(e.g. s0530, s0813, s0532, test s0802).

**Step 03 (check):** worst centroid error per case, median / max:
train 0.30 / 5.91 mm (s0818 L4), test 0.31 / 2.01 mm (s0907 L3), VerSe val 0.17 / 0.42 mm,
VerSe test 0.16 / 0.48 mm. Volume ratio preprocessed/original: train 0.68–1.35, test 0.82–1.13,
VerSe 0.98–1.01. Train: 46 cases > 1 mm, 13 > 2 mm; 45 cases with a volume ratio outside
0.9–1.1. **Nearly all outliers are TRUNCATED and/or MULTI_CC tiny partial vertebrae** (e.g. s0818
L4 = 3.3 ml cut-off piece), where a few voxels move the centre and volume a lot: not a pipeline bug.

**Side views:** orientation correct everywhere (all VerSe orientations, head up, anterior right);
L6 shown below L5 in the three L6 cases. Things to know:
- Many train crops are thorax-only (grey T12 + sliver of L1) or pelvis-only (L4/L5/sacrum).
- "Sliver" cases where the scan shows only a thin edge of the spine: s1205 (L3–L5 1–6 ml each),
  s1246 (L5 1.2 ml + sacrum), s0525 (L2 0.1 ml, L3 29 ml). Poor training examples.
- VerSe cases with only 1–2 lumbar levels: val sub-verse116, 125, 150; test sub-verse012 (ends at
  L3), 059, 119. Scoliosis makes the mid-sagittal view look jumbled: val sub-verse022, 080.
- s0573 (v3): plausible L4/L5/sacrum; the old v2 "sacrum over L3" error is gone (still METAL +
  FRAGMENT). s1221 (v3): only TRUNCATED, old fragments gone. v3 fixed both.
- Cosmetic: the page title overlaps the first row of case titles.

### Earlier (before the move)

The old laptop results and a 4-case smoke test were deleted before the move. The smoke test (steps 1-3, 4 cases each of `totalseg_v3_test` and `verse19_validation`) ran cleanly: worst centroid error 0.42 mm on the non-RAS VerSe cases, side views correctly oriented.

Findings from the old run (small v2 subset + VerSe19 val) that stay valid for VerSe:
- VerSe19 val: orientations PIR 17 / LAS 17 / PSR 6; 23 NONAXIAL; 2 ANISO (max voxel 3 mm);
  qform-only 38, sform-only 2 (no case had both set); CT stored **float32 with non-integer HU in
  38/40**, i.e. already interpolated by the provider; LABEL_AFFINE_EPS 18 (differences 2e-9–2e-7);
  NO_LUMBAR 9; one L6 case (**sub-verse023**); SPLIT_SCAN 6 scans from 3 subjects;
  lumbar coverage all L1–L5 26 / partial 5 / none 9.
- Old small-subset findings (v2 labels, now dropped): S1 overlapping L5 in 68/102 cases; s0573 =
  metal in the lumbar ROI + sacrum/S1 overlapping L3 (suspected label error); s1221 = truncated
  scan with label fragments. Check whether these recur in v3 (s0573/s1221 are also v3 cases).
- Old cache check (small subset): worst centroid error per case median 0.31 mm, max 2.88 mm.

## 7. Known issues and open TODOs

Correctness (the anomaly texts now state these honestly; the code is unchanged):
1. **NO_LUMBAR cases are still preprocessed** when they contain T12 or sacrum (step 02 only skips
   cases with none of ids 1–9). Measured: **120 train, 5 test, 1 VerSe val** such crops. Filter
   by the `labels` column of `preprocessing_status.csv` when choosing training cases, or skip them
   in step 02 (they could instead serve as negatives against issue 8). Not decided yet.
2. **NO_XFORM and BAD_PIXDIM are only flagged**, not excluded (the report says so now).
3. **Clip before blurring**: `preprocess_case_to_canonical` blurs, then `normalise_hu_to_0_1`
   clips. Padding values like −3024 HU get smeared into the outermost ring of the field of view.
   Minor; fix = clip HU (e.g. to [−1024, window max or higher]) before the Gaussian.
4. Texts starting "Plan (Part D)" describe future evaluation policy, not implemented code.

Runs (existing code):
5–7. **Done 2026-10-01** on the workstation: full steps 00–03, side views checked (see §6).

What the runs mean for the next parts (nothing implemented, wait for the user):
- **Training-case selection (Part B):** 750 train cases have all of L1–L5. Beyond NO_LUMBAR
  (issue 1), consider dropping or down-weighting cases where a lumbar level is a FRAGMENT / sliver
  (< 2 ml; e.g. s1205, s1246, s0525), and decide how to treat MULTI_CC/VOLUME_OUTLIER labels
  (label noise). MASK_OVERLAP (760) is expected (S1 vs L5) and handled by the merge order.
- **Evaluation (Part D):** in-domain test = 82 v3 test cases with ≥1 lumbar level (57 complete);
  OOD = VerSe19 36 + 31 = 67 scans with ≥1 lumbar level; report L6 cases (sub-verse023, 083, 131)
  separately. Truncated/partial vertebrae need the explicit policy from §8.
- **Part C validation:** the ctd evidence above (+11.7 mm anterior) is ready to quote in the report.

Design risks to raise with the user (Part B decisions):
8. **Crop-only training bias**: the training arrays are cropped to T12..sacrum + 30 mm, so the model
   never sees thorax/head/legs during training but will see them at full-image inference → false
   positives (e.g. a thoracic vertebra called L1). Options: bigger margin, extra random patches from
   full volumes (`--crop none`, written to a separate `_full_image` folder), or a two-stage localiser.
9. Labels are downsampled with nearest neighbour only (no anti-aliasing): jagged edges, thin
   processes can vanish; volume ratio up to 1.21 seen. The alternative is per-class soft resampling
   + argmax (~9x cost). Dice will be computed on the original grid, so this affects training labels only.
10. Each image is read twice overall (steps 01 and 02 are separate passes); `load_ct_as_hu` loads
    the full CT before cropping, and step 02 loads the CT even for cases it then skips. Fine for
    16 GB RAM; optimise only if the v3 run is too slow.
11. **Border-slice loss in `geometry.resample_onto_grid` (found 2026-10-05, NOT fixed on purpose).**
    With `mode="constant"`, sample points that land a hair outside the image (slightly tilted
    scans: direction-matrix off-diagonal ~0.07) return background, so the outermost slice can be
    dropped. Only vertebrae cut off at the scan border are affected. Worst case s0978 (L1 occupies
    4 border slices): 719 of 1366 voxels kept at 1.5 mm; with `mode="grid-constant"` all 1368.
    Shows up in step 03 as the few remaining > 1 mm / < 0.9 volume cases at 1.5 mm (10 train, 0 test).
    Also used when predictions go back to the CT grid. Left unfixed so the 2 mm vs 1.5 mm
    comparison runs identical code; the fix is one word (`grid-constant` for label/order-0 resampling)
    and should be applied to both models together, then steps 02–06 re-run.

---

## 8. Parts B–D: decisions and code (built 2026-10-01, user approved each choice)

User decisions (2026-10-01): train on the lumbar crops only (no `--crop none`, the assignment
proposes cropping); **no augmentation** beyond random patch placement (no flips: L-R flip creates
mirrored anatomy and is unnecessary because the loader makes everything RAS; S-I flip would swap
L1/L5; no simulated thick slices); use **all** usable training cases; larger batch; training
**at most 3 h** ("a short run is fine"), then use the best checkpoint; commit + push code at each
stage (user's git identity only, no Claude co-author lines); stick to what the assignment asks.

### Part B: model and training (`network.py`, `training.py`, `inference.py`, script 04)
- **Architecture:** residual-encoder U-Net (ResEnc) from Isensee et al., "nnU-Net Revisited"
  (MICCAI 2024, https://papers.miccai.org/miccai-2024/paper/2847_paper.pdf). Report argument: their
  fair benchmark (same GPU budget, all trained from scratch) found CNN U-Nets with residual encoders
  beat Transformer (SwinUNETR, nnFormer) and Mamba models; configuration matters more than novelty.
  Network code from their package `dynamic_network_architectures` (0.4.4), layout of their smallest
  preset "M": 6 stages, 32→320 channels, blocks 1-3-4-6-6-6, instance norm, leaky ReLU, no deep
  supervision; 102 M parameters; trained from scratch (no pretrained spine model).
- **Not the nnU-Net framework** (report argument): it would replace our Part A loader, chooses
  settings automatically (hard to defend in review), and mirrors along all axes by default
  (would swap L1/L5).
- **Patch 96 × 96 × 160 voxels at 2 mm (19 × 19 × 32 cm)**, chosen from the crop sizes (median
  86 × 82 × 152, 95 % within 98 × 97 × 208): one patch sees T12..sacrum → the network can count levels.
- **Training cases:** v3 train crops minus NO_LUMBAR (120) and FRAGMENT (135) → 1002 cases;
  10 % (100, seed 0) held out as validation (v3 has no val split); **902 for training**.
  List with reasons: `outputs/training/resenc_m_2mm/training_cases.csv`.
- **Patch sampling:** 2/3 centred on a random voxel of a randomly chosen present lumbar level
  (rare edge levels L1/L5 seen as often as middle ones), 1/3 uniform random.
- **Loss:** cross-entropy + soft Dice (foreground classes, batch Dice). **Optimiser = nnU-Net
  defaults unchanged** (SGD, momentum 0.99, Nesterov, lr 0.01, poly decay ^0.9, weight decay 3e-5,
  gradient clip 12), mixed precision. "Unchanged defaults" is the argument: nothing hand-tuned.
- **Batch 8** (16.5 GB on the RTX 3090). Benchmark: throughput is ~10 patches/s for batch 4–8
  (GPU already saturated at 4), batch 10 spills into shared memory and is 25× slower. So a larger
  batch did not shorten training; it only means fewer, larger steps.
- **Run:** 11 000 steps planned (= 88 000 patches ≈ 98 passes over the 902 cases), 0.72 s/step,
  validation every 1000 steps (sliding window on the 100 validation crops, Dice L1–L5),
  best checkpoint → `lumbar/weights/lumbar_vertebrae_resenc_unet.pt` (gitignored, shipped in zip),
  hard stop at 3 h. Started 2026-10-01 02:25 as a detached process (the assistant's background
  tasks are killed after 2 h): logs in `outputs/training/resenc_m_2mm/console_output.txt`,
  `training_log.csv`. Resume: `--resume`.
- **Inference** (`inference.py`): whole CT → same canonical space as training (no crop) →
  sliding window, 50 % overlap, Gaussian weighting → argmax → **largest connected piece per class**
  (a vertebra is one bone; removes e.g. a thoracic vertebra called "L1" outside the training crop)
  → nearest-neighbour back to the CT grid. The rule-based level re-ordering from the old plan was
  NOT built (keep it simple; mention as "next").

### Part C: vertebral body centre (`body_centroid.py`)
Per level in the 2 mm RAS grid: erode with a 6 mm ball (removes everything thinner than 12 mm:
pedicles ~8–10 mm, laminae, processes) → largest remaining piece that is not more than 5 mm
behind the whole-vertebra centre → dilate back 6 mm inside the original mask → centre of mass →
world RAS mm. Fallback: whole-vertebra centre (reported as `whole_vertebra_fallback`).
**Check on VerSe ground-truth masks (305 vertebrae, val + test):** body centre → VerSe point
median **0.9 mm**, 90th pct 4.0 mm, max 20 mm; whole-vertebra centre → VerSe point median
**12.4 mm** (90th 14.6). A-P offset VerSe − body median 0.4 mm. Strong evidence for the report.
Honesty note for the report: the "not behind the centre" rule was added after seeing VerSe
outliers (max 30 → 20 mm); no further tuning on VerSe was done to avoid fitting the test set.
Remaining outliers: sub-verse032 L5 (20 mm), sub-verse078 L1 (16 mm), sub-verse144 L1/L2
(metal; posterior elements apparently removed: whole centre 0.8 mm from the point), sub-verse022 L3
(scoliosis).

### Part D: evaluation (`evaluation.py`, script 06)
Rules fixed before seeing results (docstring of `evaluation.py`): Dice on the original CT grid;
level scored only where GT has it and it is not a FRAGMENT; predicted-but-absent levels counted as
false-positive levels, not as Dice 0; truncated vertebrae scored and also reported without;
shifted Dice (rename predicted levels one step up/down along T12-L1..L5-sacrum) to separate
counting errors from segmentation errors; resampling ceiling (GT → 2 mm → back); L6 patients
reported separately. Script 06 caches per case (`outputs/evaluation/per_case/`), saves predicted
labels + centroid JSONs (`outputs/predictions/<source>/`) so analysis can continue on the laptop
without a GPU (`--report-only` rebuilds the report).

### Deliverable
`lumbar/run_inference.sh <in> <out>`: creates `.venv_inference`, installs CPU torch 2.6.0 +
pinned `requirements_inference.txt`, runs script 05 on the CPU. Output per CT:
`<case>_lumbar_labels.nii.gz` (CT grid, 1–5 = L1–L5) + `<case>_centroids.json` (assignment format, RAS).

### Pitfall hit
`pip install dynamic-network-architectures` pulls `timm` → `torchvision` and **silently replaced
the CUDA torch with torch 2.14.1+cpu**. Fixed by reinstalling `torch==2.6.0 torchvision==0.21.0`
from `https://download.pytorch.org/whl/cu124` (driver 555 supports CUDA ≤ 12.5). Always check
`torch.cuda.is_available()` after installing anything.

### Results (2026-10-01) — all numbers also in `outputs/evaluation/evaluation_report.md`
- **Training:** 11 000 steps in 2.62 h; best validation mean Dice **0.897 at step 8000**
  (L1 .908, L2 .896, L3 .899, L4 .894, L5 .885); stored weights-only (391 MB). The 819 MB
  resume checkpoints stay local (`outputs/training/resenc_m_2mm/*.pt`, not pushed).
- **Dice, mean / median:** TotalSeg test (78 cases, 328 vertebrae) **0.881 / 0.928**; complete
  vertebrae only 0.908 / 0.930. VerSe19 val+test (67 cases, 305 vertebrae) **0.851 / 0.919**;
  complete only 0.854 / 0.919. Resampling ceiling 0.94 on both. Per level in the report table.
- **Part C on the test sets:** predicted body vs GT body median 0.6 mm (TS) / 0.5 mm (VerSe);
  predicted body vs VerSe point 1.0 mm (90th pct 7.5); GT body vs VerSe point 0.9 mm (3.9);
  GT whole vertebra vs VerSe point 12.5 mm. 32 of 642 predicted vertebrae fell back to the whole
  centre (small cut-off pieces).
- **Speed:** GPU evaluation median 4 s per case (the zip reading dominates, ~1 h for 189 cases);
  CPU inference 16–18 s per CT; `run_inference.sh` tested end to end in a fresh venv (CPU torch).
  On this workstation the first `pip install` is slow because its pip config lists an unreachable
  extra index (`pypi.ngc.nvidia.com`); not our problem, a fresh machine does not have it.

### Part D observations (the answers to "why the differences? anything odd?")
1. **Medians ≈ equal (0.93 vs 0.92) and near the 0.94 ceiling; the gap is in the means** → a few
   whole-case failures, not worse boundaries.
2. **Counting errors (off by one level)** in 5 cases, found by the shifted Dice: s0829 (TS, short
   chest scan), sub-verse093 (metal; 0.00 → 0.67), sub-verse078 (fractures; 0.14 → 0.79),
   sub-verse085, sub-verse260. Fix idea: rule-based level check with T12 ribs / sacrum anchors.
3. **VerSe does not label vertebrae cut off by the scan edge; TotalSegmentator does.** 12
   vertebra-sized (≥ 20 ml) "false-positive" levels on VerSe, 8 touch the scan edge; checked 5
   (sub-verse012 L4, 116 L4, 081 L3, 150 L3, 018 L5): real bone 200–360 HU, directly adjacent to
   the last labelled level, GT background. → protocol difference, not a model error. Also why
   "complete vertebrae only" changes TS (0.88 → 0.91) but not VerSe.
4. **Specks in scans without a lumbar spine**: all 27 TS test NO_LUMBAR scans get some lumbar
   label, median 0.8 ml (234 false-positive levels overall, only 16 vertebra-sized). Cause: training
   on lumbar crops only (issue 8, user chose crops). Fix idea: full-scan patches or a presence gate.
5. **L6 patients fine** (sub-verse023/083/131: 0.87–0.94 per level): the model names levels from
   the top like VerSe and leaves L6 unlabelled.
6. Other oddities for the report: VerSe CTs float with non-integer HU (pre-resampled by the
   provider); TS S1/L5 mask overlap in 817 cases (label noise at the L5/sacrum boundary);
   Part C outliers sub-verse032/078/144/022.
- Failure figure: `report/figures/failure_cases_side_views.png` (script 07): sub-verse012, s0829,
  sub-verse093, sub-verse078.

### Report
`report/lumbar_spine_report.tex` + `.pdf` (2 pages, XeLaTeX via Tectonic in the separate conda env
`latex`: `C:\Users\za76buso\AppData\Local\miniconda3\envs\latex\Library\bin\tectonic.exe`).
Pipeline diagram (TikZ, abstract stages, not modules): `report/pipeline_diagram.tex`.
Fonts are loaded by file name (texgyrepagella/heros) because Tectonic cannot find them by name.

### Git (user decision 2026-10-01)
Push everything except raw and preprocessed data: code, report, `outputs/` (qc, evaluation,
predictions, training logs), and the weights via **Git LFS** (`lumbar/weights/*.pt`, 391 MB; GitHub
rejects > 100 MB in plain git). Not pushed: `data/`, the two 819 MB resume checkpoints.

---

## 9. Retraining at 1.5 mm voxels (2026-10-05, branch `retrain-1.5mm`, user request)

**Why (user):** every TotalSegmentator v3 CT is 1.5 mm isotropic, so at 1.5 mm the training
domain loses nothing to resampling. User-measured ceiling: 2 mm TS 0.937 / VerSe 0.938;
1.5 mm TS 1.000 / VerSe 0.956 (our step 06 reproduces exactly these numbers).
**Rule:** change ONLY the voxel size; same train/validation cases, loss, optimiser, 11 000 steps,
checkpoint rule. Keep all 2 mm results; 1.5 mm results get their own names. Which model goes into
the deliverable and report is **not decided yet** (user decides after the comparison).

**Code changes (commit 42fd3d0):** voxel size is now a parameter end to end.
- `network.patch_size_for_spacing()`: patch fixed in cm (19.2 × 19.2 × 32), rounded to multiples of
  32 → 2 mm (96, 96, 160) unchanged, 1.5 mm (128, 128, 224) = 19.2 × 19.2 × 33.6 cm (213 → 224, the
  nearest multiple of 32; 192 would be only 28.8 cm).
- `sources.weights_file_for_spacing()`: 2 mm keeps `lumbar_vertebrae_resenc_unet.pt`;
  1.5 mm → `lumbar_vertebrae_resenc_unet_1.5mm.pt`.
- `training.choose_training_cases(same_cases_as=...)` + script 04 `--same-cases-as`: reuse the exact
  split of an earlier run (checked: all cases exist at the new voxel size).
- Script 04: `--spacing`, `--weights-out`; run folder `resenc_m_<s>mm`; checkpoint stores the real
  voxel size + patch (before: always 2 mm). **Refuses to overwrite an existing weights file**
  without `--overwrite-weights` (protects the 2 mm deliverable model).
- Script 06 `--tag` (picked from the checkpoint): `outputs/evaluation_1.5mm/`, `outputs/predictions_1.5mm/`.
  The 2 mm report rebuilds byte-identical. Script 07 `--tag`. New script 08 compares two evaluations.
- Already correct before: 05/06 read spacing + patch from the checkpoint; body-centre rules are in mm.

**Run facts:** step 02 at 1.5 mm: same counts as 2 mm (1257/87/32/36 written), 4.0 GB disk.
Step 03: TS centre error median 0.06–0.07 mm (2 mm: 0.3); TS test cases > 1 mm: 2 → 0; train 46 → 10
(the 10 are issue 11). RAM for training data 8.2 GB (2 mm: 3.5). **Batch 3** (15.5 GB, 0.72–0.78 s/step;
batch 4 = 20.3 GB but 3× slower per step, batch 5 overflows). → only 33 000 patches seen vs 88 000 at
2 mm (same steps): the one unavoidable difference. Training 2.59 h (`--max-hours 6` as safety only),
best validation Dice **0.905 at step 11 000** (2 mm: 0.897 at 8000; each on its own grid).

**Comparison (`outputs/evaluation_comparison_2mm_vs_1.5mm.md`, script 08):**
| | 2 mm TS | 1.5 mm TS | 2 mm VerSe | 1.5 mm VerSe |
|---|---|---|---|---|
| Dice all, mean / median | .881 / .928 | **.894 / .965** | .851 / .919 | **.906 / .939** |
| Dice not cut off | .908 / .930 | .921 / .968 | .854 / .919 | .913 / .939 |
| ceiling | .937 | 1.000 | .938 | .956 |
| miscounted cases | 1 (s0829) | 2 (s0829, s1248) | 4 (078, 085, 093, 260) | 1 (252) |
| FP levels: cases / levels / median size | 51 / 158 / 0.8 ml | 52 / 194 / 3.1 ml | 23 / 76 / 1.1 ml | 23 / 91 / 7.0 ml |
| vertebra-sized FP (≥ 20 ml) | 4 | 11 | 12 | 26 |
| body centre pred vs GT body, median / p90 | 0.6 / 8.9 | 0.3 / 15.7 | 0.5 / 7.5 | 0.6 / 3.9 |
| pred body vs VerSe point | – | – | 1.0 / 7.5 | 1.0 / 5.6 |
| CPU s/scan (6 scans, 18 cores), median | 13.5 | 33.5 (2.5×) | | |
- L5 on TS is the only level where 1.5 mm is lower in the mean (.879 → .869), due to s1248 (miscounted).
- **All extra vertebra-sized FPs are in scans with NO lumbar spine**: scans that contain a lumbar
  spine: TS 4 → 4, VerSe 9 → 10 (VerSe's unlabelled cut-off vertebrae); no-lumbar scans: TS 0 → 7,
  VerSe 3 → 16 (whole thoracic vertebrae called L1–L5, e.g. s0684, sub-verse250). Crop-only training
  weakness (issue 8), stronger at 1.5 mm; possible cause (unproven): fewer patches seen.
- TS centre p90 worse because of miscounted / FP-adjacent levels (s1248: 5 levels > 10 mm), median better.
