# Handoff: Lumbar Spine L1–L5 take-home assignment

State as of 2026-09-30. Written for the next AI assistant working on this project.
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
- Lumbar coverage in v3, **estimated** from compressed mask sizes in the zip (an empty gzipped mask
  is ~200–400 bytes; threshold 1000 bytes); confirm with the real audit:
  train 732 with all L1–L5, 1096 with ≥1 level, 734 with none;
  test 57 with all L1–L5, 79 with ≥1 level, 30 with none.
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

---

## 6. Results so far

**No current outputs exist**: the old laptop results and a 4-case smoke test were deleted before the move. The smoke test (steps 1-3, 4 cases each of `totalseg_v3_test` and `verse19_validation`) ran cleanly: worst centroid error 0.42 mm on the non-RAS VerSe cases, side views correctly oriented.

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

## 7. Known issues and open TODOs (none started; wait for the user)

Correctness (the anomaly texts now state these honestly; the code is unchanged):
1. **NO_LUMBAR cases are still preprocessed** when they contain T12 or sacrum (step 02 only skips
   cases with none of ids 1–9). With full v3 this may be hundreds of thorax/pelvis crops. Filter
   by the `labels` column of `preprocessing_status.csv` when choosing training cases, or skip them
   in step 02 (they could instead serve as negatives against issue 8).
2. **NO_XFORM and BAD_PIXDIM are only flagged**, not excluded (the report says so now).
3. **Clip before blurring**: `preprocess_case_to_canonical` blurs, then `normalise_hu_to_0_1`
   clips. Padding values like −3024 HU get smeared into the outermost ring of the field of view.
   Minor; fix = clip HU (e.g. to [−1024, window max or higher]) before the Gaussian.
4. Texts starting "Plan (Part D)" describe future evaluation policy, not implemented code.

Runs still to do (existing code):
5. Full step 01 on all four sources (≈1939 v3 + 80 VerSe cases, ≈1 h on the laptop with 3 workers; faster with more).
6. Full step 02 on all four sources (v3 train is the big one; ~4 GB estimated).
7. Full step 03, then look at the side views: s0573, s1221 (v3 labels now), PIR/PSR VerSe cases.

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

---

## 8. Agreed plan for the remaining parts (discussed with the user, nothing implemented)

**Part B, model (on the GPU workstation):** 3D U-Net (MONAI suggested for legibility; nnU-Net
rejected because it hides the decisions the user must defend in review). Input 2 mm, **128³
patches** (≈256 mm, fits T12..sacrum so the network can count levels). 8 output classes (bg,
L1–L5, T12, sacrum); report only L1–L5. Dice + cross-entropy; foreground-biased patch sampling; AMP.
Augmentation: **left–right flip only** (never flip superior–inferior or anterior–posterior; that
breaks level order), rotation ±10°, scaling 0.9–1.1, gamma, noise, and **simulated thick-slice
anisotropy** (the targeted countermeasure for VerSe; best single ablation). Train on ~200–300 v3
train cases with all L1–L5; checkpoint often. Inference: sliding window with Gaussian blending over
the full image, then a **rule-based level-consistency step** (connected components ordered along
superior–inferior, enforce monotonic labels), then resample back to the input grid. Inference must
run **CPU-only** on a fresh machine.

**Part C, vertebral body centroid:** per vertebra mask: erode with a ~5 mm ball (pedicles are
thin, ~7–10 mm; the body is ~45×30 mm) → keep the most anterior component (+y in RAS) → geodesic /
conditional dilation back inside the original mask → centre of mass → world RAS mm via the affine.
Cross-check with a second estimate (anterior half along A–P) and flag >5 mm disagreement. Validate on
TotalSegmentator GT masks first, then against VerSe ctd: expect the whole-vertebra centroid ≈12 mm
posterior of the annotation and the body centroid within a few mm. Output JSON with
`"coordinate_system": "RAS"`.

**Part D, evaluation:** in-domain = v3 official test (report n ≈ 57–79 lumbar cases);
OOD = VerSe19 validation + test. Label harmonisation table (VerSe 20–24 → L1–L5; 25 L6 and 28 T13
→ separate, cases reported separately). Dice per level on the **original image grid**. State the
empty-case policy explicitly. Report strict Dice **and** level-shift-corrected Dice (best global
shift in {−1, 0, +1}) to separate naming errors from segmentation errors. Report the **resampling
ceiling** (GT → 2 mm → back, Dice vs itself). Group split scans by subject. Expected explanations:
anisotropy, pathology/metal, field-of-view differences, label protocol differences, TotalSegmentator
label noise, transitional vertebrae, partial annotations.

**Report "what not to do":** don't chase Dice with bigger models (the ceiling is label protocol
and grid); don't train or tune on VerSe (it destroys the OOD measurement); don't build a learned body
segmenter without ground truth.

**Deliverable zip:** `lumbar/` + weights + `run_inference.sh` + pinned requirements + README
(OS, GPU optional, Docker optional). Exclude data.
