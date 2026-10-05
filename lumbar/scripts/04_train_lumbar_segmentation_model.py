"""Step 4 (Part B): train the 3D segmentation network on TotalSegmentator v3 train.

What it does
  1. chooses the training cases from step 2's crops (lumbar/training.py says which
     and why) and holds out 10 % of them as a validation set
  2. trains the ResEnc U-Net (lumbar/network.py) on random patches:
     loss = cross-entropy + Dice, SGD with momentum 0.99 and a learning rate that
     falls from 0.01 to 0 ("poly" schedule), mixed precision (float16) on the GPU.
     These are the nnU-Net default training settings, used unchanged so that none
     of them is a hand-tuned choice.
  3. every --validate-every steps: predicts the whole validation crops with the
     same sliding-window code as inference and computes Dice per level L1-L5;
     the checkpoint with the best mean Dice is kept
  4. stops after --steps, or earlier when --max-hours is reached (the best
     checkpoint so far is then the result: the assignment allows a short run)

Voxel size (--spacing, default 2 mm): the patch is fixed in centimetres
(19.2 x 19.2 x 32 cm, lumbar/network.py) and converted to voxels; the step-2 folder,
run folder and weights file carry the voxel size in their names.

Output
  lumbar/weights/lumbar_vertebrae_resenc_unet.pt       best checkpoint at 2 mm (shipped with the code)
  lumbar/weights/lumbar_vertebrae_resenc_unet_<s>mm.pt best checkpoint at another voxel size <s>
  outputs/training/<run>/last_checkpoint.pt            to resume (--resume)
  outputs/training/<run>/training_log.csv              loss per 100 steps, validation Dice
  outputs/training/<run>/training_cases.csv            every case: train / validation / excluded + why
  where <run> defaults to resenc_m_<s>mm, e.g. resenc_m_2mm, resenc_m_1.5mm.

    python scripts/04_train_lumbar_segmentation_model.py
    python scripts/04_train_lumbar_segmentation_model.py --steps 200 --validate-every 100 --run smoke_test
    python scripts/04_train_lumbar_segmentation_model.py --spacing 1.5 --batch-size 3 \\
        --same-cases-as ../outputs/training/resenc_m_1.5mm/training_cases.csv
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lumbar.inference import predict_labels_on_model_grid
from lumbar.labels import CANONICAL, TARGET_IDS
from lumbar.network import NETWORK_SETTINGS, build_lumbar_segmentation_network, patch_size_for_spacing
from lumbar.preprocess import HU_WINDOW, TARGET_SPACING
from lumbar.sources import PROJECT_ROOT, weights_file_for_spacing
from lumbar.training import (choose_training_cases, dice_per_level, dice_plus_cross_entropy,
                             load_cases_into_memory, sample_patch)

TRAINING_RUNS_DIR = PROJECT_ROOT / "outputs" / "training"


def batches_in_background(data: dict, patch_size, batch_size: int, seed: int, stop: threading.Event):
    """Prepare batches in a separate thread so the GPU never waits for numpy."""
    q = queue.Queue(maxsize=6)
    ids = list(data)

    def work():
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            pairs = [sample_patch(*data[ids[rng.integers(len(ids))]], patch_size, rng) for _ in range(batch_size)]
            x = torch.from_numpy(np.stack([p[0] for p in pairs])[:, None])
            y = torch.from_numpy(np.stack([p[1] for p in pairs]))
            while not stop.is_set():
                try:
                    q.put((x, y), timeout=1)
                    break
                except queue.Full:
                    pass

    threading.Thread(target=work, daemon=True).start()
    return q


def validate(net, val_data: dict, patch_size, device) -> dict:
    """Mean Dice per level over the validation crops (levels present in the label)."""
    net.eval()
    per_level = {c: [] for c in TARGET_IDS}
    for image, label in val_data.values():
        pred = predict_labels_on_model_grid(net, image.astype(np.float32), patch_size, device, batch_size=2)
        for c, d in dice_per_level(pred, label).items():
            per_level[c].append(d)
    net.train()
    res = {f"val_dice_{CANONICAL[c]}": float(np.mean(v)) for c, v in per_level.items() if v}
    res["val_dice_mean"] = float(np.mean(list(res.values())))
    return res


def save_checkpoint(path, net, opt, scaler, step, best, spacing, patch_size, extra, for_resuming=True):
    """for_resuming=False writes only the weights and settings (half the size: the
    optimizer state is only needed to continue training). The voxel size and patch
    size are stored, so inference and evaluation never assume them."""
    resume = {"optimizer_state": opt.state_dict(), "scaler_state": scaler.state_dict()} if for_resuming else {}
    torch.save({"model_state": net.state_dict(), **resume, "step": step, "best_val_dice": best,
                "network_settings": NETWORK_SETTINGS, "patch_size": tuple(patch_size),
                "spacing_mm": spacing, "hu_window": HU_WINDOW,
                "class_names": {i: CANONICAL.get(i, "background") for i in range(NETWORK_SETTINGS["num_classes"])},
                **extra}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacing", type=float, default=TARGET_SPACING, help="voxel size in mm; step 2 must have run with it")
    ap.add_argument("--steps", type=int, default=11000, help="planned training steps (sets the learning-rate schedule)")
    ap.add_argument("--batch-size", type=int, default=3,
                    help="patches per step (RTX 3090: 8 at 2 mm uses ~16.5 GB, 3 at 1.5 mm ~15.5 GB)")
    ap.add_argument("--max-hours", type=float, default=3.0, help="stop when this much time has passed")
    ap.add_argument("--validate-every", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--run", help="name of the folder in outputs/training/ (default resenc_m_<spacing>mm)")
    ap.add_argument("--weights-out", help="where the best checkpoint goes (default: named after the voxel size)")
    ap.add_argument("--same-cases-as", help="training_cases.csv of an earlier run: reuse its exact train/validation split")
    ap.add_argument("--overwrite-weights", action="store_true", help="allow replacing an existing weights file")
    ap.add_argument("--resume", action="store_true", help="continue from the run's last checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = patch_size_for_spacing(args.spacing)
    run_dir = TRAINING_RUNS_DIR / (args.run or f"resenc_m_{args.spacing:g}mm")
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_out = Path(args.weights_out) if args.weights_out else weights_file_for_spacing(args.spacing)
    if weights_out.exists() and not (args.resume or args.overwrite_weights):
        raise SystemExit(f"{weights_out} already exists (a trained model). Use --weights-out <new file>, "
                         "or --overwrite-weights if you really want to replace it.")
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    train_ids, val_ids, table = choose_training_cases(spacing=args.spacing, seed=args.seed,
                                                      same_cases_as=args.same_cases_as)
    table.to_csv(run_dir / "training_cases.csv", index=False)
    print(f"[train] {len(train_ids)} training cases, {len(val_ids)} validation cases; "
          f"excluded: {table['use'].str.startswith('excluded').sum()} ({table['use'].value_counts().to_dict()})"
          + (f"; split reused from {args.same_cases_as}" if args.same_cases_as else ""))
    train_data = load_cases_into_memory(train_ids, spacing=args.spacing)
    val_data = load_cases_into_memory(val_ids, spacing=args.spacing)

    net = build_lumbar_segmentation_network().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    step, best = 0, -1.0
    last_path = run_dir / "last_checkpoint.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"]); step, best = ck["step"], ck["best_val_dice"]
        print(f"[train] resumed at step {step}, best validation Dice {best:.3f}")

    log_path = run_dir / "training_log.csv"
    log = pd.read_csv(log_path).to_dict("records") if args.resume and log_path.exists() else []
    stop = threading.Event()
    batches = batches_in_background(train_data, patch_size, args.batch_size, args.seed + step, stop)
    net.train()
    t0, losses = time.time(), []
    print(f"[train] {device}, batch {args.batch_size}, spacing {args.spacing:g} mm, patch {patch_size}, steps {step}->{args.steps}, "
          f"max {args.max_hours} h")

    while step < args.steps:
        lr = args.learning_rate * (1 - step / args.steps) ** 0.9          # poly schedule
        for g in opt.param_groups:
            g["lr"] = lr
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
        step += 1

        hours = (time.time() - t0) / 3600
        out_of_time = hours >= args.max_hours
        if step % 100 == 0:
            eta = hours / step * (args.steps - step) if not args.resume else float("nan")
            print(f"   step {step}/{args.steps}  loss {np.mean(losses[-100:]):.4f}  lr {lr:.5f}  "
                  f"{hours:.2f} h  (about {eta:.2f} h left)", flush=True)
            log.append({"step": step, "loss": float(np.mean(losses[-100:])), "lr": lr, "hours": hours})
        if step % args.validate_every == 0 or step == args.steps or out_of_time:
            res = validate(net, val_data, patch_size, device)
            log.append({"step": step, "hours": hours, **res})
            msg = "  ".join(f"{k[9:]} {v:.3f}" for k, v in res.items())
            if res["val_dice_mean"] > best:
                best = res["val_dice_mean"]
                save_checkpoint(weights_out, net, opt, scaler, step, best, args.spacing, patch_size,
                                {"training_cases": len(train_ids), "validation_cases": len(val_ids)},
                                for_resuming=False)
                msg += "  -> new best, saved"
            print(f"   [validation] step {step}: {msg}", flush=True)
            save_checkpoint(last_path, net, opt, scaler, step, best, args.spacing, patch_size, {})
            pd.DataFrame(log).to_csv(log_path, index=False)
        if out_of_time:
            print(f"[train] stopped at the time limit ({args.max_hours} h) after {step} steps")
            break

    stop.set()
    print(f"[train] done: best validation mean Dice {best:.3f}; weights -> {weights_out}")


if __name__ == "__main__":
    main()
