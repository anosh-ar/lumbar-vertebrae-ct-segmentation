# Lumbar spine L1–L5 segmentation from CT

Take-home assignment: preprocessing, 3D segmentation, vertebral body centroids and
cross-dataset evaluation for the lumbar vertebrae L1–L5. The code is in `lumbar/`, the
two-page report in `report/lumbar_spine_report.pdf` (LaTeX source next to it).

The model is trained at 1.5 mm voxels, the native TotalSegmentator voxel size. Median Dice on the
test sets: 0.97 on TotalSegmentator (in-domain) and 0.94 on VerSe (out-of-domain).

## 0. Only want to run the trained model?

```bash
cd lumbar
./run_inference.sh <folder with CT .nii.gz files> <output folder>
```

Needs Python 3.10–3.12 and internet on the first run (it installs the pinned packages into
`lumbar/.venv_inference`). CPU only: no GPU, no Docker. About 1–3 min per CT on a laptop CPU (median 34 s on an 18-core workstation). Tested on Windows 11
(Git Bash); plain bash, so Linux and macOS work the same. Output per CT:
`<case>_lumbar_labels.nii.gz` (same grid as the CT, 1–5 = L1–L5) and `<case>_centroids.json`
(vertebral body centres in world mm, RAS). It uses `lumbar/weights/lumbar_vertebrae_resenc_unet_1.5mm.pt`.
The weights (`lumbar/weights/*.pt`, 391 MB each) are stored with Git LFS: run `git lfs pull` after
cloning. If Python is not found (on Windows `python3` is often only a Microsoft Store placeholder,
which the script skips), set `PYTHON=/path/to/python`. On Windows, unpack to a short folder path
(well under ~100 characters): PyTorch's DLLs fail to load beyond the 260-character path limit.

## 1. Set up (any OS, Python 3.10+, no GPU needed for steps 0–3)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e lumbar
```

## 2. Download the data by hand

The data is not in this repository. Put the zips, with exactly these names, into
`data/archives/` inside the repository folder:

| File | Source |
|---|---|
| `Totalsegmentator_dataset_v300.zip` (35 GB) | TotalSegmentator v3.0.0, Zenodo record 22688904 |
| `dataset-verse19validation.zip` | VerSe 2019, https://github.com/anjany/verse |
| `dataset-verse19test.zip` | VerSe 2019, https://github.com/anjany/verse |

Do not unzip the TotalSegmentator file; the code reads it in place. To keep the data somewhere
else, set `LUMBAR_DATA=/path/to/data` (that folder must contain `archives/`).

## 3. Run the pipeline

Run from inside `lumbar/`. With no `--source`, each step processes every downloaded dataset
(TotalSegmentator v3 train + test, VerSe19 validation + test). Every step can be re-run safely.
On a machine with more cores and RAM, raise `--workers` (each worker holds one CT in memory).

```bash
cd lumbar
python scripts/00_extract_verse_archives.py                        # unzip VerSe into data/raw/
python scripts/01_find_header_and_label_anomalies.py --workers 8   # -> outputs/qc/anomaly_report.md
python scripts/02_preprocess_cases_for_training.py --workers 8     # -> data/preprocessed/*_1.5mm_lumbar_crop/
python scripts/03_verify_preprocessed_cases.py                     # -> outputs/qc/preprocessing_check_*
```

Add `--limit 5` to steps 01, 02 and 06 for a quick test. Each script's `--help` explains what it does
and what it writes.

Parts B–D (training needs an NVIDIA GPU: install the CUDA build of PyTorch first, e.g.
`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`;
check `torch.cuda.is_available()` afterwards, because installing other packages can replace it with
the CPU build):

```bash
python scripts/04_train_lumbar_segmentation_model.py --max-hours 6 \
    --same-cases-as ../outputs/training/resenc_m_1.5mm/training_cases.csv
                                                      # Part B: 2.6 h on an RTX 3090 (batch 3) -> weights/..._1.5mm.pt
python scripts/05_predict_lumbar_vertebrae.py <in> <out>   # Parts B+C on any CT folder (what run_inference.sh calls)
python scripts/06_evaluate_on_test_sets.py                 # Part D: -> outputs/evaluation_1.5mm/evaluation_report.md
python scripts/07_make_report_figures.py --tag 1.5mm       # failure-case figure for the report
python scripts/09_draw_body_centre_steps.py                # body-centre figure for the report (Part C)
```

To rebuild the report PDF: `cd report && pdflatex lumbar_spine_report.tex` (any LaTeX with TikZ).
