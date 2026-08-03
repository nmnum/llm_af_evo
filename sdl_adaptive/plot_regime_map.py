"""
plot_regime_map.py — Visualise batch size and pool composition results
as a regime map: landscape property × parameter → AUC.

Run after ablation_batch_pool.py completes both studies.

Usage:
    python plot_regime_map.py --out_dir results_ablation2/
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REGIME_ORDER = ["flat", "smooth", "structured", "multimodal"]
REGIME_COLOURS = {
    "flat":       "#F44336",
    "smooth":     "#2196F3",
    "structured": "#4CAF50",
    "multimodal": "#FF9800",
}


def plot_study1(df, out_dir):
    """Batch size effect by regime."""
    if "batch_size" not in df.columns:
        print("batch_size column not found — run study 1 first")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: AUC vs batch_size, lines per regime
    ax = axes[0]
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime].dropna(subset=["batch_size"])
        if sub.empty:
            continue
        bs_vals = sorted(sub["batch_size"].unique())
        means = [sub[sub["batch_size"]==bs]["auc_best"].mean() for bs in bs_vals]
        sems  = [sub[sub["batch_size"]==bs]["auc_best"].sem()  for bs in bs_vals]
        ax.errorbar(bs_vals, means, yerr=sems, label=regime,
                    color=REGIME_COLOURS[regime], marker="o", linewidth=2)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean AUC")
    ax.set_title("AUC vs Batch Size by Landscape Regime")
    ax.legend()
    ax.set_xticks([1, 2, 4, 8])

    # Right: joint vs refit, for batch>1
    ax2 = axes[1]
    sub_multi = df[df["batch_size"] > 1].dropna(subset=["refit"])
    if not sub_multi.empty:
        for regime in REGIME_ORDER:
            sub_r = sub_multi[sub_multi["regime"] == regime]
            if sub_r.empty:
                continue
            joint_mean = sub_r[~sub_r["refit"]]["auc_best"].mean()
            refit_mean = sub_r[ sub_r["refit"]]["auc_best"].mean()
            ax2.scatter(["Joint (no refit)", "Sequential (refit)"],
                        [joint_mean, refit_mean],
                        color=REGIME_COLOURS[regime], s=100, label=regime)
            ax2.plot(["Joint (no refit)", "Sequential (refit)"],
                     [joint_mean, refit_mean],
                     color=REGIME_COLOURS[regime], alpha=0.5)

    ax2.set_ylabel("Mean AUC")
    ax2.set_title("Joint vs Sequential Selection by Regime")
    ax2.legend()

    plt.tight_layout()
    path = out_dir / "batch_size_regime_map.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # Print the key table for the paper
    print("\nBATCH SIZE × REGIME TABLE:")
    pivot = df.groupby(["regime","batch_size"])["auc_best"].mean().unstack("batch_size")
    print(pivot.round(3).to_string())


def plot_study2(df, out_dir):
    """Pool composition effect: ea_ratio × pool_size heatmap."""
    if "ea_ratio" not in df.columns:
        print("ea_ratio column not found — run study 2 first")
        return

    regimes = [r for r in REGIME_ORDER if r in df["regime"].values]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5*len(regimes), 4))
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = df[df["regime"] == regime].dropna(subset=["ea_ratio","pool_size"])
        if sub.empty:
            continue
        pivot = sub.groupby(["pool_size","ea_ratio"])["auc_best"].mean().unstack("ea_ratio")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=df["auc_best"].quantile(0.1),
                       vmax=df["auc_best"].quantile(0.9))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.0%}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(int))
        ax.set_xlabel("EA candidate ratio")
        ax.set_ylabel("Pool size")
        ax.set_title(f"{regime}")
        plt.colorbar(im, ax=ax, label="AUC")

        # Mark best cell
        best_idx = np.unravel_index(np.argmax(pivot.values), pivot.values.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=3))

    fig.suptitle("Pool Composition Regime Map\n"
                 "(black border = best configuration)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "pool_composition_regime_map.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # Acquisition function comparison
    if "acq" in df.columns:
        print("\nACQUISITION FUNCTION × REGIME:")
        pivot_acq = df.groupby(["regime","acq"])["auc_best"].mean().unstack("acq")
        print(pivot_acq.round(3).to_string())

    # Key finding: does acquisition matter when pool is random?
    rand_acq = df[df["acq"].str.startswith("rand", na=False)]
    ucb_acq  = df[df["acq"] == "ucb_2"]
    if not rand_acq.empty and not ucb_acq.empty:
        t, p = stats.ttest_ind(ucb_acq["auc_best"], rand_acq["auc_best"])
        print(f"\nUCB_2 vs Random acquisition: "
              f"Δ={ucb_acq['auc_best'].mean()-rand_acq['auc_best'].mean():+.3f} "
              f"p={p:.3f}" + (" *" if p<0.05 else ""))
        if p > 0.05:
            print("  → Acquisition function doesn't matter much: pool quality dominates")
        else:
            print("  → Acquisition function matters: scoring quality is important")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results_ablation2")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)

    # Load both study results if available
    for study_csv, plot_fn in [
        ("batch_size_summary.csv",                 plot_study1),
        ("pool_composition_frac_summary.csv",       plot_study2),
        ("pool_composition_full_summary.csv",       plot_study2),
    ]:
        csv_path = out_dir / study_csv
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"\nLoaded: {csv_path} ({len(df)} rows)")
            plot_fn(df, out_dir)
        else:
            print(f"Not found (run the study first): {csv_path}")


if __name__ == "__main__":
    main()
