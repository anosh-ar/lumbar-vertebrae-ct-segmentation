# Lumbar Spine L1–L5 take-home assignment

**Read `HANDOFF.md` before doing anything.** It holds the assignment summary, data facts, a code
map, results so far, open issues and the agreed plan.

Standing rules from the user:
- Do not write or change code until the user explicitly says to continue. Explaining, answering,
  moving files and running existing scripts to answer a question are fine.
- Explain things in plain, easy language; docstrings and reports too.
- Names of scripts, functions and generated files must say what they do (3-5 words is fine).
- The project moved to a stronger workstation, which runs the whole pipeline from step 00
  (data downloaded there by hand). Inference must still work CPU-only on a fresh machine.
- Git commits are authored by the user only: no Claude co-author or attribution lines.
- Raw and preprocessed data (`data/`) never go into git. Since 2026-10-01 (user decision) code,
  report, `outputs/` and the weights (via Git LFS) are pushed; resume checkpoints are not (`.gitignore`).
- Never extract `data/archives/Totalsegmentator_dataset_v300.zip` (35 GB); the code reads it in place.

Quick facts:
- Python: a `.venv` at the repo root with `pip install -e lumbar` (setup steps in `README.md`).
- Scripts live in `lumbar/scripts/`, numbered in pipeline order:
  `00_extract_verse_archives`, `01_find_header_and_label_anomalies`,
  `02_preprocess_cases_for_training`, `03_verify_preprocessed_cases`,
  `04_train_lumbar_segmentation_model`, `05_predict_lumbar_vertebrae`,
  `06_evaluate_on_test_sets`, `07_make_report_figures`. Report: `report/lumbar_spine_report.pdf`.
- Workstation: conda env `EMGNet` (CUDA torch 2.6.0); see HANDOFF.md §2.
  With no `--source`, every script processes all downloaded full datasets
  (TotalSegmentator v3 train + test from the zip, VerSe19 validation + test).
- The code deliverable is the `lumbar/` folder; data lives in `data/`, reports in `outputs/qc/`.
- On the old Windows laptop the project path contains spaces: always quote paths.
