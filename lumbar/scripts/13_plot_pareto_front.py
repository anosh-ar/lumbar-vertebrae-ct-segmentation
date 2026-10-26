"""Step 13 (NAS extension): Pareto front of the search, and the model to retrain.

What it does
  1. reads step 12's table (outputs/nas/search_trials.csv)
  2. finds the Pareto front: architectures that no other one beats in BOTH Dice
     (higher) and CPU seconds per CT (lower)
  3. plots Dice against CPU seconds, front connected, baseline (the deliverable's
     architecture) marked; writes a small table (Dice, CPU time, parameters, memory)
  4. picks the architecture to retrain fully: the FASTEST point of the front whose
     search Dice is at most --dice-margin below the baseline's search Dice
     (rule fixed before looking at the results; --pick overrides it). If that is the
     baseline itself, the fallback takes the best Dice among the Pareto points at
     least 2x faster than the baseline (fallback added during the overnight run)
  With --min-speedup S (second search, rule fixed before it started): only Pareto
  points at least S times faster than the baseline count; the fastest of them within
  --dice-margin is picked, if none is: the one with the best Dice.

Output (outputs/nas/, or --out-dir)
  pareto_front_dice_vs_cpu_time.png
  pareto_front.md                 table of every evaluated architecture, front marked
  chosen_architecture.json        input for script 04 --architecture

    python scripts/13_plot_pareto_front.py
    python scripts/13_plot_pareto_front.py --pick 3mm_5stages_w16_M
    python scripts/13_plot_pareto_front.py --out-dir ../outputs/nas_1.5mm --min-speedup 1.5 --dice-margin 0.03
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lumbar.architecture_search_space import BASELINE_ARCHITECTURE, architecture_name
from lumbar.sources import NAS_OUTPUTS_DIR


def pareto_front(df: pd.DataFrame) -> pd.Series:
    """True for rows that no other row beats in both objectives (>= Dice and <= time, one strictly)."""
    def dominated(r):
        better = (df["dice_mean"] >= r["dice_mean"]) & (df["cpu_seconds_per_case"] <= r["cpu_seconds_per_case"])
        strictly = (df["dice_mean"] > r["dice_mean"]) | (df["cpu_seconds_per_case"] < r["cpu_seconds_per_case"])
        return bool((better & strictly).any())
    return ~df.apply(dominated, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dice-margin", type=float, default=0.02, help="allowed Dice loss vs the baseline for the pick")
    ap.add_argument("--pick", help="architecture name to retrain instead of the rule")
    ap.add_argument("--min-speedup", type=float, help="only Pareto points at least this many times faster "
                    "than the baseline can be picked (see above)")
    ap.add_argument("--out-dir", default=str(NAS_OUTPUTS_DIR), help="folder of the search (step 12 --out-dir)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    df = pd.read_csv(out_dir / "search_trials.csv").drop_duplicates("architecture").reset_index(drop=True)
    df["pareto"] = pareto_front(df)
    base_name = architecture_name(BASELINE_ARCHITECTURE)
    base = df.loc[df["architecture"] == base_name].iloc[0]

    if args.pick:
        chosen, rule = df.loc[df["architecture"] == args.pick].iloc[0], "--pick"
    elif args.min_speedup:
        fast = df[df["pareto"] & (df["cpu_seconds_per_case"] <= base["cpu_seconds_per_case"] / args.min_speedup)]
        if fast.empty:
            raise SystemExit(f"no Pareto point is {args.min_speedup}x faster than the baseline")
        ok = fast[fast["dice_mean"] >= base["dice_mean"] - args.dice_margin]
        if not ok.empty:
            chosen = ok.sort_values("cpu_seconds_per_case").iloc[0]
            rule = (f"fastest Pareto point at least {args.min_speedup:g}x faster than the baseline "
                    f"and within {args.dice_margin} Dice of it")
        else:
            chosen = fast.sort_values("dice_mean").iloc[-1]
            rule = (f"no Pareto point at least {args.min_speedup:g}x faster is within {args.dice_margin} Dice; "
                    f"the best Dice among them")
    else:
        ok = df[df["pareto"] & (df["dice_mean"] >= base["dice_mean"] - args.dice_margin)]
        chosen = ok.sort_values("cpu_seconds_per_case").iloc[0]
        rule = f"fastest Pareto point within {args.dice_margin} Dice of the baseline"
        if chosen["architecture"] == base_name:
            # fallback, added after the first trials of the overnight run: when no faster model is
            # close to the baseline, retraining the baseline would show nothing new
            fast = df[df["pareto"] & (df["cpu_seconds_per_case"] <= base["cpu_seconds_per_case"] / 2)]
            if not fast.empty:
                chosen = fast.sort_values("dice_mean").iloc[-1]
                rule = ("no faster model within the margin; fallback: best Dice among the Pareto points "
                        "at least 2x faster than the baseline")
    arch = {k: chosen[k] for k in ("spacing_mm", "n_stages", "base_width", "blocks")}
    arch = {"spacing_mm": float(arch["spacing_mm"]), "n_stages": int(arch["n_stages"]),
            "base_width": int(arch["base_width"]), "blocks": str(arch["blocks"])}
    (out_dir / "chosen_architecture.json").write_text(json.dumps(arch, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for spacing, marker in zip(sorted(df["spacing_mm"].unique()), "os^"):
        d = df[df["spacing_mm"] == spacing]
        ax.scatter(d["cpu_seconds_per_case"], d["dice_mean"], marker=marker, s=40, color="0.6",
                   label=f"{spacing:g} mm voxels")
    front = df[df["pareto"]].sort_values("cpu_seconds_per_case")
    ax.plot(front["cpu_seconds_per_case"], front["dice_mean"], "-o", color="tab:blue", label="Pareto front")
    ax.scatter([base["cpu_seconds_per_case"]], [base["dice_mean"]], marker="*", s=250, color="tab:red",
               zorder=5, label="baseline (deliverable architecture)")
    ax.scatter([chosen["cpu_seconds_per_case"]], [chosen["dice_mean"]], s=200, facecolors="none",
               edgecolors="tab:green", linewidths=2, zorder=6, label="chosen for full retraining")
    # numbers next to the front points and a key with the names: close points stay readable
    for i, (_, r) in enumerate(front.iterrows(), 1):
        ax.annotate(str(i), (r["cpu_seconds_per_case"], r["dice_mean"]), fontsize=8, fontweight="bold",
                    xytext=(5, -12 if i % 2 else 5), textcoords="offset points", color="tab:blue")
    key = "\n".join(f"{i}  {r['architecture']}" for i, (_, r) in enumerate(front.iterrows(), 1))
    ax.text(0.98, 0.03, "Pareto front:\n" + key, transform=ax.transAxes, fontsize=7, ha="right", va="bottom", multialignment="left",
            family="monospace", bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9})
    ax.set_xscale("log")
    ax.set_xlabel("CPU seconds per CT (median of 3 CTs, log scale)")
    ax.set_ylabel(f"validation Dice L1-L5 after {int(df['steps'].iloc[0])} training steps")
    ax.set_title("Architecture search: Dice vs CPU time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_front_dice_vs_cpu_time.png", dpi=150)

    cols = ["architecture", "dice_mean", "cpu_seconds_per_case", "parameters_millions", "peak_ram_gb",
            "gflops_per_patch", "training_minutes", "pareto"]
    table = df.sort_values("cpu_seconds_per_case")[cols].round(3)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    text = (f"# Architecture search: every evaluated architecture\n\n"
            f"Dice: validation Dice L1-L5 on the original CT grid after the short search training.\n"
            f"Baseline: {base_name}. Chosen for full retraining: **{chosen['architecture']}**"
            f" (rule: {rule}).\n\n"
            + "\n".join(lines) + "\n")
    (out_dir / "pareto_front.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
