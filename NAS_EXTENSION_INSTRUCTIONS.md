# Instructions for the workstation assistant: NAS extension

Written 2026-10-26 on the laptop for the AI code assistant on the workstation.
Read all of it before you start. It is self-contained: `CLAUDE.md` and `HANDOFF.md` are no
longer in git (see "Before you start").

---

## 1. What the user wants

The user is applying for a research-associate job (Visual Computing, Uni Siegen, DFG project on
**multi-criteria, hardware-aware neural architecture search, optimising sensor and network
parameters together**). To prepare and to show hands-on experience, they want a **small,
practical extension** of this project, around **one weekend of work**:

> Hardware-aware, multi-objective architecture search for the lumbar segmentation model:
> find smaller/faster models for **CPU-only inference** that keep nearly the same Dice.

This is **not** a research study and **not** a paper. Keep it simple and practical. Each method
should still be a known, published technique (cited in the README section). No extra research
questions, benchmark tables or ablations beyond what is described here.

**This note is the user's go-ahead to write code for this extension**, on the new branch only.
Still ask the user before: installing packages, starting a large download, pushing, or anything
that touches `main`.

---

## 2. Standing rules from the user (keep following them)

- Explain in plain, easy language: messages, docstrings, README text. The user writes quickly;
  typos are normal. Give honest answers, including "this is not done / did not work".
- Names of scripts, functions and generated files must say what they do (3-5 words is fine).
- Write code that reads like the existing code (same comment style, naming, idiom).
- **Git commits are authored by the user only** (their git identity). **No Claude co-author or
  attribution lines** in commits or PRs.
- `data/` never goes into git. Never extract `data/archives/Totalsegmentator_dataset_v300.zip`
  (35 GB); the code reads it in place.
- Inference must still work CPU-only on a fresh machine.
- Branches: `main` = finished take-home deliverable (1.5 mm model), `model-2mm` = first model,
  frozen. **Do not change either. Do not merge anything into them.**
- After installing anything, check `torch.cuda.is_available()`: installing
  `dynamic-network-architectures` once silently replaced the CUDA torch with a CPU build. Fix:
  `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`.

---

## 3. Before you start

1. **Save the local notes first.** The latest commit on `main` (`502150a`) deletes `CLAUDE.md`
   and `HANDOFF.md` from git (they are now in `.gitignore`), so `git pull` deletes them from the
   folder too. If the workstation has its own copies, back them up before pulling. To get the
   last committed versions back after the pull:
   `git show 502150a~1:CLAUDE.md > CLAUDE.md` and `git show 502150a~1:HANDOFF.md > HANDOFF.md`
   (they stay untracked). `HANDOFF.md` has the full project history if you need more detail.
2. `git checkout main && git pull`, then `git fetch origin` and
   `git checkout nas-efficiency` (the branch already exists on GitHub; this note is its first
   commit). Work only on this branch.
3. Python: conda env **`EMGNet`** (CUDA torch 2.6.0). conda is not on PATH in the assistant's
   shell; call `C:\Users\za76buso\AppData\Local\miniconda3\envs\EMGNet\python.exe` directly.
   Repo was at `D:\Users\za76buso\temp\lumbar-spine-segmentation` (check).
   Hardware: i9-10980XE (18 cores), 128 GB RAM, RTX 3090 24 GB, Windows 11.
4. **New package needed: `optuna`** (for the NSGA-II search). Ask the user before
   `pip install optuna` into `EMGNet`, then check torch CUDA is still there. Optuna is a training-
   time tool only: do **not** add it to `lumbar/requirements_inference.txt`.

---

## 4. Data: nothing new to download

The extension uses the **same data** as the existing pipeline. No new dataset is needed.

Check that these exist (the pipeline has already run on the workstation, so they should):
- `data/archives/Totalsegmentator_dataset_v300.zip` (35 GB, TotalSegmentator v3.0.0, Zenodo
  record 22688904)
- `data/archives/dataset-verse19validation.zip`, `dataset-verse19test.zip` (VerSe 2019,
  https://github.com/anjany/verse), extracted by script 00
- preprocessed training crops at 1.5 mm and 2 mm under `data/preprocessed/`
  (`<source>_<spacing>mm_lumbar_crop`)

**What is new:** crops at **3 mm**. Make them with the existing script:
`python lumbar/scripts/02_preprocess_cases_for_training.py --spacing 3` (then script 03 with
`--spacing 3` to check them). Pass the source of the training set if needed; see `--help`.

**If any archive is missing:** try to download it (ask the user before starting the 35 GB
TotalSegmentator download). If the download fails or needs a login/manual step, **stop and tell
the user to download it by hand** into `data/archives/`, with the source links above.

---

## 5. Project background (what exists)

Lumbar spine L1-L5 segmentation from CT (take-home assignment). Code is the `lumbar/` package
(`pip install -e lumbar`); scripts in `lumbar/scripts/`, numbered 00-10 in pipeline order. Read
`README.md` for the full overview.

Current model (`lumbar/src/lumbar/network.py`):
- Residual-encoder U-Net (ResEnc, Isensee et al., "nnU-Net Revisited", MICCAI 2024), from the
  package `dynamic_network_architectures`; preset "M": 6 stages, features 32-64-128-256-320-320,
  encoder blocks 1-3-4-6-6-6, decoder 1 conv per stage, instance norm, leaky ReLU, no deep
  supervision, 102 M parameters. 8 output classes (background, L1-L5, T12, sacrum).
- Settings live in `NETWORK_SETTINGS`; `build_lumbar_segmentation_network(settings)` builds any
  variant. Checkpoints store `network_settings`, `patch_size`, `spacing_mm`;
  `load_trained_network()` rebuilds from the checkpoint, so inference and script 06 already work
  for any architecture **if the checkpoint stores the right settings**.
- Patch is fixed in cm (19.2 x 19.2 x 32), rounded to multiples of 32 voxels
  (`patch_size_for_spacing`): 2 mm -> 96x96x160, 1.5 mm -> 128x128x224.

Training (`lumbar/scripts/04_train_lumbar_segmentation_model.py`, `lumbar/src/lumbar/training.py`):
- 902 TotalSegmentator v3 train crops for training, 100 held out for validation (seed 0; list in
  `outputs/training/<run>/training_cases.csv`, reuse with `--same-cases-as`).
- Loss CE + soft Dice; nnU-Net default optimiser unchanged (SGD, momentum 0.99, Nesterov,
  lr 0.01, poly decay, weight decay 3e-5, grad clip 12), mixed precision. No augmentation beyond
  random patch placement (on purpose: flips would swap left/right or L1/L5).
- 1.5 mm: batch 3 (15.5 GB GPU), 0.72-0.78 s/step, 11 000 steps = 2.6 h. Best val Dice 0.905.
- Validation inside script 04 is on the crops in the model grid.

Results of the deliverable model (1.5 mm, ResEnc-M), for comparison:
- Dice mean / median: TotalSegmentator test 0.894 / 0.965, VerSe19 0.906 / 0.939.
- **CPU time: median 33.5 s per scan** on the workstation (6 scans, 18 cores); 2 mm model 13.5 s.
  Fresh laptop: 45-164 s per scan.

---

## 6. The extension to build

### Idea (with the sources to cite in the README)
| Part | What | Source |
|---|---|---|
| Search space | voxel spacing (the "sensor" parameter) + depth + width + blocks | EfficientNet (Tan & Le, ICML 2019): scale depth, width and input resolution together |
| Hardware cost | **measured** CPU time per case, not FLOPs | MnasNet (Tan et al., CVPR 2019): measure latency on the real device |
| Search | Optuna `NSGAIISampler`, two objectives | NSGA-II (Deb et al., IEEE TEVC 2002); NSGA-Net (Lu et al., GECCO 2019) |
| Cheap evaluation | each candidate trains for the same short budget; best ones are retrained fully | standard low-fidelity NAS practice |

Check each reference (authors, venue, year) before writing it into the README.

### Search space (54 configurations)
- spacing: {1.5, 2, 3} mm
- n_stages: {4, 5, 6}
- base width: {8, 16, 32}; features double per stage, capped at 320
  (e.g. 16 -> 16-32-64-128-256-320)
- encoder blocks: "light" = 1-1-2-2-2-2, "M" = 1-3-4-6-6-6 (first n_stages entries)
- strides [1, 2, 2, ...] (n_stages entries), decoder 1 conv per stage (n_stages-1 entries)
- 3 mm patch: 192/3 = 64, 320/3 = 106.7 -> rounds to 96 (28.8 cm). The current rounding to
  multiples of 32 is fine for fewer stages too (32 is a multiple of 8 and 16). Mention the 3 mm
  patch size in the README.
- Add the current model (1.5 mm, 6 stages, width 32, M) to the search as a fixed first trial
  (`study.enqueue_trial`), so the baseline is measured with the same protocol.

### Objectives
1. **Validation Dice L1-L5 on the original CT grid** (maximise). Different spacings are only
   comparable on the original grid: predict a fixed subset of ~30 validation cases with
   `inference.predict_ct` (same path as real inference) and compare with their ground truth from
   the zip. The validation inside script 04 (model grid) is NOT comparable across spacings.
2. **CPU time per case** (minimise): `predict_ct` on 3 fixed CTs (earlier tests used s1446,
   sub-verse020, sub-verse053), median of 3 runs, CPU only, fixed thread count
   (`torch.set_num_threads`, e.g. 8; write it down). Include resampling, as real inference does.
- Also record (not optimised): parameter count, peak RAM, and GFLOPs if easy.

### Steps
1. **Measure CPU time of all 54 configurations** with untrained weights (no training needed;
   network + resampling time). Fast, gives the full cost table. Optional: drop configurations
   slower than the baseline.
2. **Search:** Optuna NSGA-II, ~30 trials (population ~10, i.e. ~3 generations). Each trial:
   build network -> train a **fixed short budget, same for every trial** (~1 500 steps, ~20 min;
   same batch size, lr schedule over the short budget) -> objectives above. Use SQLite storage
   (`outputs/nas/...db`) so the study can resume after a crash. Sequential runs are fine
   (~10 h overnight); run trials as a detached process, the assistant's own background tasks get
   killed after ~2 h.
3. **Retrain 2 Pareto models fully** (e.g. "fastest with acceptable Dice" and "balanced"; let
   the user choose from the Pareto plot) with script 04 and the full 11 000-step schedule.
4. **Evaluate** them with script 06 on both test sets, then compare with the deliverable model
   (script 08 can compare two evaluations).
5. **Plot** the Pareto front: Dice vs CPU seconds per case, baseline marked; small results table
   (Dice, CPU time, parameters, memory).
6. **README section** on the branch (about half a page, plain language): what was searched, how,
   sources, results, limitations (small search space, short training is only a proxy, one CPU).

### Suggested files (names say what they do)
- `lumbar/src/lumbar/architecture_search_space.py`: configuration -> network settings + patch
- `lumbar/scripts/11_measure_cpu_time_of_architectures.py`
- `lumbar/scripts/12_search_architectures_with_nsga2.py`
- `lumbar/scripts/13_plot_pareto_front.py`
- script 04: new `--architecture <json>` option, so full retrains reuse it

### Pitfalls found while reading the code
- `save_checkpoint` in script 04 stores the global `NETWORK_SETTINGS`, not the network that was
  trained. It must store the **actual** settings, or the checkpoint loads the wrong architecture.
- Script 06 picks its output folder from the voxel size when `--tag` is not given. A 1.5 mm NAS
  model would then **overwrite the deliverable's `outputs/evaluation_1.5mm/`**. Always pass an
  explicit `--tag` (e.g. `nas_fast`).
- Script 04 refuses to overwrite existing weights; keep it that way. Give NAS weights their own
  names.
- `.gitignore` and Git LFS include `lumbar/weights/lumbar_vertebrae_resenc_unet*.pt`. A NAS
  weights file named like that would be committed via LFS. Put NAS weights somewhere else (e.g.
  `outputs/nas/weights/`) or ask the user which ones should be pushed.
- Training data at 1.5 mm takes 8.2 GB RAM per process; fine on 128 GB, but keep it in mind if
  you run trials in parallel. ResEnc-M at 1.5 mm, batch 3 uses 15.5 GB GPU, so only small
  models fit two at a time.
- pandas: `df.flags` is a built-in attribute; use `df["flags"]`.
- Very long bash heredocs with Python code failed to parse; write files with the file tool.

### Optional (only if time is left; ask the user)
- **SynFlow** (Tanaka et al., NeurIPS 2020; as a NAS proxy: Abdelfattah et al., ICLR 2021) as
  a side check: compute it and the parameter count for every configuration and check if their
  ranking matches the short-training Dice. Instance norm breaks SynFlow on an all-ones input;
  switch norm layers off for the computation and say so.
- **ONNX Runtime** export of the chosen fast model and a CPU speed comparison with PyTorch.

---

## 7. Plan for the weekend

- Saturday: data check + 3 mm preprocessing, search space module, step 1 (CPU cost table),
  search script, smoke test with 2 short trials; start the search overnight.
- Sunday: user picks 2 Pareto models; full retrains; evaluation; plot; README section.
- Commit on `nas-efficiency` at each working stage (user's identity, no co-author lines). Ask
  before pushing.
