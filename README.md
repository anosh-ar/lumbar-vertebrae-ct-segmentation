# Lumbar spine L1–L5 segmentation from CT

Take-home assignment: preprocessing, 3D segmentation, vertebral body centroids and
cross-dataset evaluation for the lumbar vertebrae L1–L5. The code is in `lumbar/`.
Part A (preprocessing and its checks) is written; Parts B–D are planned in `HANDOFF.md`.

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
python scripts/02_preprocess_cases_for_training.py --workers 8     # -> data/preprocessed/
python scripts/03_verify_preprocessed_cases.py                     # -> outputs/qc/preprocessing_check_*
```

Add `--limit 5` to any step for a quick test. Each script's `--help` explains what it does
and what it writes.
