# Lumbar spine L1–L5 segmentation from CT

This project finds the five lumbar vertebrae (L1–L5) in a CT scan and the centre of each
vertebral body. For every CT it writes:

- a label map: every voxel marked as background, L1, L2, L3, L4 or L5;
- a JSON file with the centre of each vertebral body in world coordinates (RAS, mm).

A 3D U-Net, trained from scratch on TotalSegmentator, does the segmentation. A shape rule then
cuts the vertebral body out of each vertebra. Then a hardware-aware architecture search (NAS)
looked for a smaller U-Net that runs faster on a CPU. It found a model with 5 resolution stages
and 16 channels in the first stage. That model is **2x faster on the CPU and 3.7x smaller, with
almost the same Dice**.

![Example segmentations](report/figures/example_segmentations.png)

*Model output on four test scans (side view, head up, front of the body to the right): predicted
L1 (red) to L5 (blue), and the predicted centre of each vertebral body (white cross).*

**Results** (Dice per vertebra, 1 = perfect overlap with the ground truth; median / mean):

| Test set | ResEnc M U-Net (102 M parameters) | Searched U-Net (28 M parameters) |
|---|---|---|
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) test split (in-domain, 78 scans) | **0.97** / 0.89 | **0.96** / 0.90 |
| [VerSe 2019](https://github.com/anjany/verse) (out-of-domain, never seen in training, 67 scans) | **0.94** / 0.91 | **0.94** / 0.90 |
| CPU time per CT (median of 6 scans) | 32 s | **15 s** |

Body centres lie a median of **1.0 mm** from the centre points marked by VerSe experts, with
both models. Most of the remaining errors come from a few scans where whole vertebrae get the
wrong level name.

The two-page report [`report/lumbar_spine_report_Rezaei.pdf`](report/lumbar_spine_report_Rezaei.pdf)
covers the ResEnc M model (sections 1–4). The architecture search is described in section 5.

![Pipeline overview](report/figures/pipeline_overview.png)

*Preprocessing is shared by training (top) and inference (bottom). The evaluation compares
the outputs with the ground truth on the original CT grid.*

## Run the trained model

```bash
cd lumbar
./run_inference.sh <folder with CT files> <output folder>
```

- Input: CT files (`.nii.gz` or `.nii`) in any orientation.
- Output per CT: `<case>_lumbar_labels.nii.gz` (same grid as the CT, 1–5 = L1–L5) and
  `<case>_centroids.json`.
- Needs Python 3.10–3.12 and internet on the first run, which installs the pinned packages into
  `lumbar/.venv_inference`. CPU only: no GPU, no Docker. Tested on Windows 11 (Git Bash); plain bash,
  so Linux and macOS work the same. About 1–3 min per CT on a laptop.
- The weights (`lumbar/weights/*.pt`, 391 MB) are stored with Git LFS: run `git lfs pull` after cloning.
- If Python is not found, set `PYTHON=/path/to/python`. On Windows, keep the folder path short
  (well under 100 characters): PyTorch's DLLs fail to load past the 260-character path limit.
- `run_inference.sh` uses the ResEnc M weights. The searched model runs with
  `python scripts/05_predict_lumbar_vertebrae.py <in> <out> --weights ../outputs/nas/weights/1.5mm_5stages_w16_M.pt --device cpu`.
  Its weights (112 MB) are not in git.

## 1. Preprocessing and data problems

**Data.** TotalSegmentator v3 (1939 CTs) with its official split: 1830 for training, 109 for
testing. VerSe 2019 (80 CTs) is used only for testing, so it stays an unseen dataset. One loader
reads both.

**Three steps, the same in training and inference:**

1. **Orientation:** image and labels are turned into RAS orientation through their affine matrix.
2. **Voxel size:** 1.5 mm in every direction. This is the native size of all TotalSegmentator
   scans, so the training data are not degraded.
3. **Intensity:** Hounsfield units are clipped to −1000…1500 and scaled to 0…1. A fixed window keeps
   bone at the same value in every scan.

**Data problems.** An automatic check tests every file for 30 kinds of problems and decides how
each is handled. The most common: VerSe images not stored axially (48 of 80), overlapping
TotalSegmentator masks at S1/L5 (817 scans), vertebrae cut off by the scan edge (431), metal near
the spine (124 + 4), and three VerSe patients with a sixth lumbar vertebra (L6). The full table is
in the report and in `outputs/qc/anomaly_report.md`. Preprocessing moves structure centres by a
median of 0.07 mm (TotalSegmentator) and 0.46 mm (VerSe).

## 2. Model and training

- **Network:** the residual-encoder U-Net ("ResEnc M", 102 M parameters) from *nnU-Net Revisited*
  (Isensee et al., MICCAI 2024), trained from scratch. Only the network is taken, not the nnU-Net
  framework: its default mirroring along every axis would swap L1 and L5.
- **Classes:** L1–L5, plus T12 and the sacrum as landmarks that help count the levels.
- **Training data:** crops from T12 to the sacrum + 3 cm; 902 training and 100 validation scans.
  Each patch is 128 × 128 × 224 voxels (19 × 19 × 34 cm), big enough to hold the whole lumbar spine.
- **Setup:** cross-entropy + Dice loss, nnU-Net default optimiser settings unchanged, no data
  augmentation, 11 000 steps, batch 3, 2.6 h on one RTX 3090. Best validation Dice: 0.905.
- **Inference:** the network slides over the whole scan; per level only the largest connected piece
  is kept, and the result is mapped back to the original CT grid.

ResEnc M is the starting point (baseline) for the architecture search in section 5. The searched
model uses the same data, training cases, loss and schedule.

## 3. Vertebral body centre

The body is the only part of a vertebra thicker than about 12 mm. So the method:

1. shrinks the vertebra mask by 6 mm, which removes all thin parts;
2. keeps the largest remaining piece that does not lie behind the vertebra's centre;
3. grows it back by 6 mm, but only inside the original mask;
4. takes its centre of mass in world coordinates.

![Body centre steps](report/figures/body_centre_steps.png)

*L3 of VerSe scan sub-verse020, front at the top. Green: kept; red: removed.*

**Check against VerSe.** VerSe marks one point per vertebra, and these points sit at the body
centre. On the ground-truth masks the extracted centre lies a median of **0.9 mm** from them,
against 12.5 mm for the centre of the whole vertebra. On the predicted masks: 1.0 mm.

## 4. Evaluation on two test sets

Dice is computed on the original CT grid, only for levels present in the ground truth. VerSe
L6 and T13 keep their own labels, so they are never counted as L5, T12 or background. The numbers
below are for ResEnc M. Section 5 compares the searched models on the same test sets.

| Test set | L1 | L2 | L3 | L4 | L5 | All | Limit |
|---|---|---|---|---|---|---|---|
| TotalSegmentator test (in-domain, 78 scans) | .88 / .97 | .90 / .97 | .92 / .97 | .91 / .97 | .87 / .96 | **.89 / .97** | 1.00 |
| VerSe19 (out-of-domain, 67 scans) | .90 / .94 | .91 / .94 | .92 / .94 | .91 / .94 | .89 / .93 | **.91 / .94** | .96 |

*Mean / median Dice. Limit: the best score possible at 1.5 mm (ground truth resampled to 1.5 mm and back).*

**Why the numbers differ.** VerSe voxels do not lie on the 1.5 mm grid, which alone caps its
Dice at 0.96. Measured against these caps, a typical vertebra is segmented about equally well on
both datasets. The means are lower than the medians because a few scans fail as a whole:

- **Wrong level count:** in 3 scans the vertebrae are found but named one level off: short scans
  without the sacrum or ribs as landmarks, or next to metal.
- **Cut-off vertebrae** are hard to segment and name. TotalSegmentator labels them, VerSe hardly
  does; without them the TotalSegmentator mean rises from 0.89 to 0.92.
- **False lumbar vertebrae** in scans that show only the chest or neck: 23 whole vertebrae are
  labelled L1–L5. The model was trained on lumbar crops only and never saw such anatomy.
- **Patients with L6** are handled correctly (Dice 0.87–0.95 per level).

![Failure cases](report/figures/failure_cases_side_views_1.5mm.png)

*Ground truth next to prediction: (a) cut-off vertebra unlabelled in VerSe, (b, c) levels shifted
by one, (d) neck and chest scan labelled L1–L5. L1 red, L2 orange, L3 yellow, L4 green, L5 blue,
T12 grey, sacrum purple.*

## 5. Architecture search for a faster CPU model

**Goal.** The model has to run on a CPU, where ResEnc M needs about 32 s per CT. Can a smaller
network be clearly faster and keep nearly the same Dice? A small multi-objective architecture
search answers this, so the network size is not guessed.

**Search space** (`lumbar/src/lumbar/architecture_search_space.py`):

- voxel size 1.5 or 3 mm: how finely the CT is resampled;
- 4, 5 or 6 resolution stages: the depth;
- 8, 16 or 32 channels in the first stage: the width, which doubles per stage up to at most 320;
- "light" (1-1-2-2-2-2) or "M" (1-3-4-6-6-6) residual blocks per stage.

Changing resolution, depth and width together follows EfficientNet [1]. This gives 36
architectures; ResEnc M (1.5 mm, 6 stages, width 32, M) is one of them. The patch always covers
19.2 x 19.2 x 32 cm. At 3 mm it is 64 x 64 x 96 voxels (28.8 cm long, rounded to a multiple of 32).

**Procedure.**

1. *CPU cost, measured* (script 11). This is the full CPU inference of `predict_ct` (resampling,
   sliding window, clean-up) with untrained weights, on 3 fixed CTs and 8 threads, for all 36
   architectures. As in MnasNet [2], the cost is measured on the real machine, not estimated from
   FLOPs. Width and voxel size set the cost; depth and block count hardly matter
   (`outputs/nas/cpu_time_per_architecture.csv`).
2. *Search* (script 12) uses Optuna's [3] NSGA-II sampler [4], the genetic algorithm also used by
   NSGA-Net [5], with two objectives:
   - maximise Dice L1–L5 of 20 fixed validation CTs, measured on the **original CT grid**, the
     only grid where 1.5 and 3 mm models are comparable;
   - minimise CPU seconds per CT, from step 1.

   Every candidate trains from scratch for the same short budget (same batch, optimiser and
   schedule as script 04). The population is 6, and ResEnc M is the first trial.
3. *Pick, retrain, evaluate* (scripts 13, 04, 06, 08). The pick rule is fixed before each round.
   The picked architecture is retrained with the full 11 000-step schedule on the same training
   cases as ResEnc M, then evaluated on both test sets like it.

The search ran in two rounds.

**Round 1: 1.5 and 3 mm, 1500 steps per candidate.** It tried 12 architectures in 2 h. The model
it picked (3 mm, 4 stages, width 32, light) is 9x faster, but its test Dice is about 0.1 lower.
Most of this loss comes from the voxel size itself, not from the small network: even a perfect
3 mm prediction cannot score more than 0.89 (TotalSegmentator) and 0.91 (VerSe) on the original
grid. So round 2 kept the 1.5 mm voxels. The Pareto plot of round 1 is
`outputs/nas/pareto_front_dice_vs_cpu_time.png`.

**Round 2: 1.5 mm only, 3000 steps per candidate.** It tried 12 of the 18 architectures in 5 h.
The longer budget is needed because after 1500 steps most narrow models had not started to
segment yet. Pick rule: among the Pareto points at least 1.5x faster than ResEnc M, take the
fastest one within 0.03 Dice of it; if there is none, take the one with the best Dice. No fast
point came within 0.03 Dice after the short training, so the rule picked **5 stages, width 16, M
blocks**. For comparison, the point with the best search Dice (5 stages, width 32, light, only
1.2x faster) was also fully retrained.

![Pareto front, round 2](outputs/nas_1.5mm/pareto_front_dice_vs_cpu_time.png)

| Search Dice after 3000 steps | CPU s per CT (search table) | Architecture (1.5 mm) |
|---|---|---|
| 0.842 | 21.0 | 6 stages, w32, M (ResEnc M) |
| 0.858 | 17.2 | 5 stages, w32, light (retrained for comparison) |
| 0.856 | 16.9 | 6 stages, w32, light |
| 0.374 | 9.7 | 5 stages, w16, M (picked) |
| 0.327 / 0.293 | 8.4 / 8.3 | 6 / 5 stages, w16, light |
| 0.07–0.15 | 4.8–5.2 | width 8 (4 architectures) |

Full table: `outputs/nas_1.5mm/pareto_front.md`.

**Result: all retrained models side by side.** The test sets are as in section 4. CPU time was
measured for all four models in one session on the same 6 scans (i9-10980XE, 18 cores,
`outputs/cpu_inference_seconds_per_scan.csv`). Parameters and peak RAM come from script 11.

| | ResEnc M (baseline) | **Searched: 1.5 mm, w16, M** | 1.5 mm, w32, light | 3 mm, w32, light (round 1) |
|---|---|---|---|---|
| Architecture | 1.5 mm, 6 stages, w32, M | 1.5 mm, 5 stages, w16, M | 1.5 mm, 5 stages, w32, light | 3 mm, 4 stages, w32, light |
| Parameters | 102.4 M | **28.0 M** | 24.3 M | 9.5 M |
| CPU s per scan, median of 6 (speed-up) | 32.0 | **15.0 (2.1x)** | 25.5 (1.25x) | 3.5 (9x) |
| Peak RAM, CPU inference (8 threads) | 10.3 GB | **5.8 GB** | 10.0 GB | 1.9 GB |
| Validation Dice (100 cases, model grid) | 0.905 | 0.913 | 0.915 | 0.90 |
| Dice TotalSeg test, mean / median | 0.894 / 0.965 | **0.900 / 0.962** | 0.896 / 0.968 | 0.788 / 0.882 |
| Dice TotalSeg test, not cut off, mean | 0.921 | 0.923 | 0.937 | 0.833 |
| Dice VerSe19, mean / median | 0.906 / 0.939 | **0.895 / 0.937** | 0.891 / 0.939 | 0.828 / 0.893 |
| Best possible Dice at this voxel size, TotalSeg / VerSe | 1.000 / 0.956 | 1.000 / 0.956 | 1.000 / 0.956 | 0.891 / 0.913 |
| Scans counted one level off, TotalSeg / VerSe | 2 / 1 | 1 / 1 | 1 / 1 | 3 / 2 |
| Body centre to ground truth, 90th pct, TotalSeg / VerSe (mm) | 15.7 / 3.9 | 6.3 / 4.0 | 13.9 / 4.0 | 26.9 / 8.3 |
| Vertebra-sized false-positive levels (>= 20 ml), TotalSeg / VerSe | 11 / 26 | 26 / 21 | 8 / 12 | 8 / 15 |

**The result of the search is the 1.5 mm, 5-stage, width-16, M model.** On the CPU it is 2x
faster than ResEnc M (15 s vs 32 s per scan), needs about half the memory and has 3.7x fewer
parameters. Its test Dice is almost the same: 0.900 vs 0.894 mean on TotalSegmentator and 0.895
vs 0.906 on VerSe, with medians within 0.003. It counts levels at least as well, and its body
centres are closer to the ground truth on TotalSegmentator and as close on VerSe. Its one weak point is more large
false-positive pieces on TotalSegmentator: 26 vs 11 levels of at least 20 ml predicted where the
ground truth has none. The other two models mark the ends of the trade-off. The width-32 light
model has the best Dice on complete vertebrae and the fewest false positives, but it is only
1.25x faster. The 3 mm model is 9x faster but 0.1 Dice lower. Full tables:
`outputs/evaluation_comparison_deliverable_vs_nas_3mm_w32_light_vs_nas_1.5mm_w16_M_vs_nas_1.5mm_w32_light.md`.

**Limitations.**
- The short search training is only a rough stand-in for the full training, and it underrates
  narrow models badly. The picked model scored 0.374 in the search but 0.913 validation Dice after
  the full training, as good as the wide models. A search that trains each candidate longer would
  probably move the Pareto front towards smaller models.
- One training run per model: Dice differences of about 0.01 between the 1.5 mm models could
  change with another random seed.
- In round 1 no faster model was within the planned 0.02 Dice. A fallback (best Dice among
  models at least 2x faster) was added during the run.
- Small search: 12 architectures per round, about 2.5 generations of 6. 2 mm voxels were not
  searched.
- One CPU (i9-10980XE). The cost table of step 1 has one run per CT, and its peak RAM includes
  the three loaded CTs.
- The 3 mm training crops are coarser: volume ratio preprocessed / original 0.87–1.30 (script 03),
  against 0.95–1.00 at 1.5 mm.
- The searched weights (`outputs/nas/weights/`, 3 files, 237 MB) are not in git.

**Files.**
- Round 1, `outputs/nas/`: CPU-time table, search table (`search_trials.csv`), Optuna study
  (`architecture_search.db`), Pareto plot and table (`pareto_front.md`), chosen architecture,
  console logs.
- Round 2, `outputs/nas_1.5mm/`: the same files, plus `driver_log.txt`.
- Evaluations: `outputs/evaluation_nas_1.5mm/` (the searched model),
  `outputs/evaluation_nas_1.5mm_w32_light/` and `outputs/evaluation_nas_fast/` (3 mm). The matching
  `predictions_*` and `training/*` folders are next to them.

**Sources.**
[1] M. Tan, Q. V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.
[2] M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, Q. V. Le. MnasNet: Platform-Aware Neural Architecture Search for Mobile. CVPR 2019.
[3] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama. Optuna: A Next-generation Hyperparameter Optimization Framework. KDD 2019.
[4] K. Deb, A. Pratap, S. Agarwal, T. Meyarivan. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2), 2002.
[5] Z. Lu, I. Whalen, V. Boddeti, Y. Dhebar, K. Deb, E. Goodman, W. Banzhaf. NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm. GECCO 2019.

## 6. Next steps

- Train with some patches from whole scans, or add a "lumbar spine present?" check, against the
  false vertebrae. This matters most for the searched model, which has more of them.
- Check the level names with a rule (vertebrae ordered head to feet, T12 and sacrum as anchors).
- Train longer: the best checkpoint was the last one.
- Search with a longer training budget per candidate and include 2 mm voxels; then use the
  searched model in `run_inference.sh`.

## Reproduce the whole pipeline

**Setup** (any OS, Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e lumbar
```

**Data** (not in this repository). Put these zips into `data/archives/`, or set
`LUMBAR_DATA=/path/to/data` to a folder that contains `archives/`:

| File | Source |
|---|---|
| `Totalsegmentator_dataset_v300.zip` (35 GB, do not unzip) | TotalSegmentator v3.0.0, Zenodo record 22688904 |
| `dataset-verse19validation.zip`, `dataset-verse19test.zip` | VerSe 2019, https://github.com/anjany/verse |

**Steps** (run from inside `lumbar/`; each script's `--help` says what it does and writes):

```bash
python scripts/00_extract_verse_archives.py                       # unzip VerSe
python scripts/01_find_header_and_label_anomalies.py --workers 8  # data problems -> outputs/qc/anomaly_report.md
python scripts/02_preprocess_cases_for_training.py --workers 8    # training crops at 1.5 mm
python scripts/03_verify_preprocessed_cases.py                    # checks + side views -> outputs/qc/
python scripts/04_train_lumbar_segmentation_model.py --max-hours 6 \
    --same-cases-as ../outputs/training/resenc_m_1.5mm/training_cases.csv    # needs an NVIDIA GPU
python scripts/05_predict_lumbar_vertebrae.py <in> <out>          # what run_inference.sh calls
python scripts/06_evaluate_on_test_sets.py                        # Dice, centres -> outputs/evaluation_1.5mm/
python scripts/07_make_report_figures.py --tag 1.5mm              # failure-case figure
python scripts/09_draw_body_centre_steps.py                       # body-centre figure
python scripts/10_draw_example_segmentations.py                   # example figure at the top of this README
# architecture search (section 5; needs `pip install optuna`)
python scripts/02_preprocess_cases_for_training.py --spacing 3 --workers 8   # 3 mm crops
python scripts/11_measure_cpu_time_of_architectures.py            # CPU time of all architectures
# round 1: 1.5 and 3 mm
python scripts/12_search_architectures_with_nsga2.py --max-hours 2 # NSGA-II search -> outputs/nas/
python scripts/13_plot_pareto_front.py                            # Pareto plot, chosen_architecture.json
python scripts/04_train_lumbar_segmentation_model.py --architecture ../outputs/nas/chosen_architecture.json \
    --same-cases-as ../outputs/training/resenc_m_1.5mm/training_cases.csv --run nas_fast
python scripts/06_evaluate_on_test_sets.py --weights ../outputs/nas/weights/3mm_4stages_w32_light.pt --tag nas_fast
# round 2: 1.5 mm only
python scripts/12_search_architectures_with_nsga2.py --spacings 1.5 --steps 3000 --trials 12 --population 6 \
    --validation-cases 20 --max-hours 6 --study lumbar_cpu_nas_1.5mm --out-dir ../outputs/nas_1.5mm
python scripts/13_plot_pareto_front.py --out-dir ../outputs/nas_1.5mm --min-speedup 1.5 --dice-margin 0.03
python scripts/04_train_lumbar_segmentation_model.py --architecture ../outputs/nas_1.5mm/chosen_architecture.json \
    --same-cases-as ../outputs/training/resenc_m_1.5mm/training_cases.csv --run nas_1.5mm
python scripts/06_evaluate_on_test_sets.py --weights ../outputs/nas/weights/1.5mm_5stages_w16_M.pt --tag nas_1.5mm
python scripts/04_train_lumbar_segmentation_model.py --architecture ../outputs/nas_1.5mm/second_choice_architecture.json \
    --same-cases-as ../outputs/training/resenc_m_1.5mm/training_cases.csv --run nas_1.5mm_w32_light
python scripts/06_evaluate_on_test_sets.py --weights ../outputs/nas/weights/1.5mm_5stages_w32_light.pt --tag nas_1.5mm_w32_light
python scripts/08_compare_two_evaluations.py --tags 1.5mm nas_fast nas_1.5mm nas_1.5mm_w32_light \
    --names deliverable nas_3mm_w32_light nas_1.5mm_w16_M nas_1.5mm_w32_light
```

Add `--limit 5` to steps 01, 02 and 06 for a quick test. For training, install the CUDA build of
PyTorch first (`pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`)
and check `torch.cuda.is_available()` afterwards: installing other packages can replace it with
the CPU build.

## Datasets

- **TotalSegmentator** v3.0.0 — https://github.com/wasserth/TotalSegmentator (data: Zenodo record 22688904).
  Wasserthal J. et al., *TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images*,
  Radiology: Artificial Intelligence, 2023.
- **VerSe 2019** — https://github.com/anjany/verse.
  Sekuboyina A. et al., *VerSe: A Vertebrae labelling and segmentation benchmark for multi-detector CT images*,
  Medical Image Analysis, 2021.

**Where things are:** code in `lumbar/` (library in `lumbar/src/lumbar/`, scripts in
`lumbar/scripts/`), results in `outputs/`, report and figures in `report/`. The first model,
trained at 2 mm, is kept on the branch `model-2mm`.
