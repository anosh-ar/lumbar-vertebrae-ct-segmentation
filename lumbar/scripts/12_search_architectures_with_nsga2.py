"""Step 12 (NAS extension): search architectures for CPU inference with NSGA-II.

Two objectives, both on things a user of the model sees
  1. Dice L1-L5 on the ORIGINAL CT grid (maximise), on a fixed subset of the
     validation cases of the deliverable's split. Models with different voxel sizes
     can only be compared there: the Dice inside script 04 is on the model grid.
  2. CPU seconds per CT (minimise), measured on the real machine by step 11 and
     read from outputs/nas/cpu_time_per_architecture.csv (so training on the GPU never
     disturbs the timing).

How
  Optuna's NSGAIISampler (NSGA-II: Deb et al., IEEE TEVC 2002; used for NAS in
  NSGA-Net: Lu et al., GECCO 2019) proposes architectures from the search space
  (lumbar/architecture_search_space.py). The first trial is the deliverable's
  architecture, so the baseline goes through the same protocol.
  Every trial trains its network from scratch for the SAME short budget (--steps,
  same batch size, poly learning-rate schedule over those steps, nnU-Net optimiser
  of script 04). A short training is only a cheap stand-in for the full one
  ("low-fidelity" evaluation); the best models are then retrained fully with
  script 04 --architecture.
  An architecture that was already evaluated is not trained again: the trial
  gets the stored result. The search stops after --trials different architectures
  or --max-hours.

Output (outputs/nas/, or --out-dir for another search, e.g. outputs/nas_1.5mm/)
  architecture_search.db       Optuna study (SQLite): a crashed search resumes from it
  search_trials.csv            one row per evaluated architecture: Dice per level, CPU s, size
  search_console_output.txt    when started detached (see README)

    python scripts/12_search_architectures_with_nsga2.py
    python scripts/12_search_architectures_with_nsga2.py --spacings 1.5 --steps 3000 --out-dir ../outputs/nas_1.5mm
    python scripts/12_search_architectures_with_nsga2.py --trials 2 --steps 50 --validation-cases 2 --study smoke_test
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import TrialState

from lumbar.architecture_search_space import (BASELINE_ARCHITECTURE, BASE_WIDTHS, ENCODER_BLOCKS, N_STAGES,
                                              SEARCHED_SPACINGS_MM, architecture_name, network_settings_for,
                                              patch_size_for)
from lumbar.geometry import resample_onto_grid
from lumbar.inference import ct_to_model_grid, predict_labels_on_model_grid
from lumbar.labels import CANONICAL, TARGET_IDS
from lumbar.network import build_lumbar_segmentation_network
from lumbar.preprocess import merge_label_files_to_canonical
from lumbar.sources import NAS_OUTPUTS_DIR, PROJECT_ROOT, load_cases
from lumbar.training import (batches_in_background, choose_training_cases, dice_per_level,
                             dice_plus_cross_entropy, load_cases_into_memory)

CPU_TIME_TABLE = NAS_OUTPUTS_DIR / "cpu_time_per_architecture.csv"     # shared by every search
DELIVERABLE_SPLIT = PROJECT_ROOT / "outputs" / "training" / "resenc_m_1.5mm" / "training_cases.csv"


class ValidationCTs:
    """The fixed validation CTs with their ground truth on the original grid.
    The CT on the model grid only depends on the voxel size, so it is computed once
    per voxel size and kept (same functions as inference.predict_ct)."""
    def __init__(self, case_ids):
        by_id = {c.case_id: c for c in load_cases("totalseg_v3_train")}
        self.cases = {}
        for cid in case_ids:
            img = by_id[cid].load_ct()
            gt, _ = merge_label_files_to_canonical(by_id[cid], img)
            self.cases[cid] = (img, gt)
        self._model_grid = {}

    def on_model_grid(self, cid, spacing):
        if (cid, spacing) not in self._model_grid:
            self._model_grid[cid, spacing] = ct_to_model_grid(self.cases[cid][0], spacing)
        return self._model_grid[cid, spacing]

    def dice_on_original_grid(self, net, patch_size, spacing, device) -> dict:
        """Dice per level L1-L5, averaged over every (case, level present) pair."""
        net.eval()
        per_level = {c: [] for c in TARGET_IDS}
        for cid, (img, gt) in self.cases.items():
            image, affine = self.on_model_grid(cid, spacing)
            label_model = predict_labels_on_model_grid(net, image, patch_size, device, batch_size=4)
            pred = resample_onto_grid(label_model, affine, img.affine, img.shape[:3], order=0)
            for c, d in dice_per_level(pred, gt).items():
                per_level[c].append(d)
        res = {f"dice_{CANONICAL[c]}": float(np.mean(v)) for c, v in per_level.items() if v}
        res["dice_mean"] = float(np.mean([d for v in per_level.values() for d in v]))
        return res


def train_short_budget(net, train_data, patch_size, steps, batch_size, learning_rate, seed, device):
    """Same training as script 04 (loss, optimiser, poly schedule, mixed precision),
    for a fixed number of steps and without validation in between."""
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    stop = threading.Event()
    batches = batches_in_background(train_data, patch_size, batch_size, seed, stop)
    net.train()
    losses, t0 = [], time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = learning_rate * (1 - step / steps) ** 0.9
        x, y = batches.get()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = dice_plus_cross_entropy(net(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            print(f"      step {step + 1}/{steps}  loss {np.mean(losses[-200:]):.4f}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    stop.set()
    return float(np.mean(losses[-100:])), (time.time() - t0) / 60


def suggest_architecture(trial, spacings) -> dict:
    return {"spacing_mm": float(trial.suggest_categorical("spacing_mm", list(spacings))),
            "n_stages": trial.suggest_categorical("n_stages", list(N_STAGES)),
            "base_width": trial.suggest_categorical("base_width", list(BASE_WIDTHS)),
            "blocks": trial.suggest_categorical("blocks", list(ENCODER_BLOCKS))}


def write_trials_table(study, path):
    rows = [{"trial": t.number, **t.user_attrs, "dice_mean": t.values[0], "cpu_seconds_per_case": t.values[1]}
            for t in study.get_trials(states=(TrialState.COMPLETE,)) if "repeat_of" not in t.user_attrs]
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=12, help="different architectures to evaluate (baseline included)")
    ap.add_argument("--population", type=int, default=6, help="NSGA-II population size")
    ap.add_argument("--steps", type=int, default=800, help="training steps per trial (same for every trial)")
    ap.add_argument("--batch-size", type=int, default=3, help="same for every trial (3 fits the baseline at 1.5 mm)")
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--validation-cases", type=int, default=20, help="fixed validation CTs for the Dice objective")
    ap.add_argument("--spacings", type=float, nargs="+", default=list(SEARCHED_SPACINGS_MM))
    ap.add_argument("--same-cases-as", default=str(DELIVERABLE_SPLIT), help="train/validation split to use")
    ap.add_argument("--max-hours", type=float, default=10.0, help="no new trial starts after this time")
    ap.add_argument("--study", default="lumbar_cpu_nas", help="study name inside the SQLite file")
    ap.add_argument("--out-dir", default=str(NAS_OUTPUTS_DIR),
                    help="folder for the study, the trials table and logs (one folder per search)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_table = out_dir / "search_trials.csv"
    cpu_table = pd.read_csv(CPU_TIME_TABLE).set_index("architecture")
    t_start = time.time()

    split = {s: choose_training_cases(spacing=s, same_cases_as=args.same_cases_as)[:2] for s in args.spacings}
    train_ids, val_ids = split[args.spacings[0]]
    val_subset = sorted(val_ids)[:args.validation_cases]
    print(f"[search] {len(train_ids)} training cases; Dice on {len(val_subset)} validation CTs (original grid)")
    val_cts = ValidationCTs(val_subset)
    train_data = {}                                     # voxel size -> crops in RAM, loaded when first needed

    storage = f"sqlite:///{(out_dir / 'architecture_search.db').as_posix()}"
    study = optuna.create_study(study_name=args.study, storage=storage, load_if_exists=True,
                                directions=["maximize", "minimize"],
                                sampler=optuna.samplers.NSGAIISampler(population_size=args.population, seed=args.seed))
    for t in study.get_trials(states=(TrialState.RUNNING,)):     # left over from a crash
        study.tell(t.number, state=TrialState.FAIL)
    if not study.trials and BASELINE_ARCHITECTURE["spacing_mm"] in args.spacings:
        study.enqueue_trial(BASELINE_ARCHITECTURE)

    def evaluated():
        return {t.user_attrs["architecture"]: t for t in study.get_trials(states=(TrialState.COMPLETE,))
                if "repeat_of" not in t.user_attrs}

    def objective(trial):
        arch = suggest_architecture(trial, args.spacings)
        name = architecture_name(arch)
        before = evaluated()
        if name in before:                              # NSGA-II proposed it again: reuse the result
            trial.set_user_attr("architecture", name)
            trial.set_user_attr("repeat_of", before[name].number)
            print(f"   trial {trial.number}: {name} already evaluated (trial {before[name].number})")
            return tuple(before[name].values)
        s = arch["spacing_mm"]
        if s not in train_data:
            train_data[s] = load_cases_into_memory(split[s][0], spacing=s)
        settings, patch = network_settings_for(arch), patch_size_for(arch)
        cpu_seconds = float(cpu_table.loc[name, "cpu_seconds_per_case"])
        print(f"   trial {trial.number}: {name}  patch {patch}  CPU {cpu_seconds:.1f} s/CT", flush=True)
        torch.manual_seed(args.seed)
        net = build_lumbar_segmentation_network(settings).to(device)
        loss, minutes = train_short_budget(net, train_data[s], patch, args.steps, args.batch_size,
                                           args.learning_rate, args.seed, device)
        res = val_cts.dice_on_original_grid(net, patch, s, device)
        del net
        torch.cuda.empty_cache()
        for k, v in {"architecture": name, **arch, "final_loss": loss, "training_minutes": minutes,
                     "steps": args.steps, **res,
                     **cpu_table.loc[name, ["parameters_millions", "gflops_per_patch", "peak_ram_gb"]].to_dict()}.items():
            trial.set_user_attr(k, v)
        print(f"   trial {trial.number}: {name}  Dice {res['dice_mean']:.3f}  CPU {cpu_seconds:.1f} s/CT  "
              f"({minutes:.1f} min training)", flush=True)
        return res["dice_mean"], cpu_seconds

    attempts = 0
    while len(evaluated()) < args.trials and attempts < 5 * args.trials:
        if (time.time() - t_start) / 3600 > args.max_hours:
            print(f"[search] time limit of {args.max_hours} h reached")
            break
        study.optimize(objective, n_trials=1)
        write_trials_table(study, trials_table)
        attempts += 1
    print(f"[search] {len(evaluated())} architectures evaluated in {(time.time() - t_start) / 3600:.2f} h; "
          f"table -> {trials_table}")


if __name__ == "__main__":
    main()
