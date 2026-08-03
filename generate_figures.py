"""
generate_figures.py — Generate publication-quality figures from benchmark results.

Produces:
  1. HV trajectory by condition (line plot with confidence bands)
  2. Experiments-to-90%-HV bar chart by condition
  3. Final HV bar chart by condition (grouped by protein/prior)
  4. Sample efficiency curve (HV vs experiment count)
  5. IGD comparison bar chart

Usage:
    python generate_figures.py --results /mnt/results/benchmark_phase1/phase1_raw_results.csv
"""

import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc_params
import seaborn as sns

# Font and style settings
rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'
rc_params['figure.dpi'] = 150
rc_params['savefig.dpi'] = 150

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "mo_random": "#999999",
    "mo_egbo": "#0072B2",
    "mo_egbo_real": "#56B4E9",
    "mo_egbo_novelty": "#009E73",
    "mo_ls_na_egbo": "#D55E00",
    "mo_llm_candidate_gen": "#CC79A7",
    "mo_llm_existing": "#E69F00",
}

LABELS = {
    "mo_random": "Random",
    "mo_egbo": "EGBO (light)",
    "mo_egbo_real": "EGBO (BoTorch)",
    "mo_egbo_novelty": "NA-EGBO",
    "mo_ls_na_egbo": "LS-NA-EGBO",
    "mo_llm_candidate_gen": "LLM-CandGen",
    "mo_llm_existing": "LLM (existing)",
}

ORDER = ["mo_random", "mo_egbo", "mo_egbo_real", "mo_egbo_novelty",
         "mo_ls_na_egbo", "mo_llm_candidate_gen"]


def fig_final_hv_grouped(df, out_path):
    """Bar chart of final HV by condition, grouped by protein/prior."""
    fig, axes = plt.subplots(1, len(df.groupby(["protein", "prior_level"])),
                             figsize=(5 * len(df.groupby(["protein", "prior_level"])), 5),
                             sharey=True)
    if len(df.groupby(["protein", "prior_level"])) == 1:
        axes = [axes]

    for ax, ((protein, prior), group) in zip(axes, df.groupby(["protein", "prior_level"])):
        conds = [c for c in ORDER if c in group["condition"].unique()]
        means = [group[group["condition"] == c]["final_hv"].mean() for c in conds]
        stds = [group[group["condition"] == c]["final_hv"].std() for c in conds]
        colors = [COLORS.get(c, "#333333") for c in conds]
        labels = [LABELS.get(c, c) for c in conds]

        bars = ax.bar(range(len(conds)), means, yerr=stds, color=colors,
                      capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{protein}\n{prior} prior", fontsize=10)
        ax.set_ylabel("Final Hypervolume" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Final Hypervolume by Condition", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_exp_to_90pct(df, out_path):
    """Bar chart of experiments needed to reach 90% of max HV."""
    fig, axes = plt.subplots(1, len(df.groupby(["protein", "prior_level"])),
                             figsize=(5 * len(df.groupby(["protein", "prior_level"])), 5),
                             sharey=True)
    if len(df.groupby(["protein", "prior_level"])) == 1:
        axes = [axes]

    for ax, ((protein, prior), group) in zip(axes, df.groupby(["protein", "prior_level"])):
        conds = [c for c in ORDER if c in group["condition"].unique()]
        means = [group[group["condition"] == c]["exp_to_90pct_hv"].mean() for c in conds]
        stds = [group[group["condition"] == c]["exp_to_90pct_hv"].std() for c in conds]
        colors = [COLORS.get(c, "#333333") for c in conds]
        labels = [LABELS.get(c, c) for c in conds]

        ax.bar(range(len(conds)), means, yerr=stds, color=colors,
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{protein}\n{prior} prior", fontsize=10)
        ax.set_ylabel("Experiments to 90% HV" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        # Lower is better
        ax.invert_yaxis()  # so shorter bars appear "better" at top

    plt.suptitle("Sample Efficiency: Experiments to 90% of Max HV (lower = better)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_igd_comparison(df, out_path):
    """Bar chart of IGD (lower = closer to true Pareto front = better)."""
    fig, axes = plt.subplots(1, len(df.groupby(["protein", "prior_level"])),
                             figsize=(5 * len(df.groupby(["protein", "prior_level"])), 5),
                             sharey=True)
    if len(df.groupby(["protein", "prior_level"])) == 1:
        axes = [axes]

    for ax, ((protein, prior), group) in zip(axes, df.groupby(["protein", "prior_level"])):
        conds = [c for c in ORDER if c in group["condition"].unique()]
        means = [group[group["condition"] == c]["igd"].mean() for c in conds]
        stds = [group[group["condition"] == c]["igd"].std() for c in conds]
        colors = [COLORS.get(c, "#333333") for c in conds]
        labels = [LABELS.get(c, c) for c in conds]

        ax.bar(range(len(conds)), means, yerr=stds, color=colors,
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{protein}\n{prior} prior", fontsize=10)
        ax.set_ylabel("IGD (lower = better)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Inverted Generational Distance (lower = closer to true Pareto front)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_hv_boxplot(df, out_path):
    """Box plot of final HV by condition, faceted by protein/prior."""
    conds_present = [c for c in ORDER if c in df["condition"].unique()]
    df_filtered = df[df["condition"].isin(conds_present)].copy()
    df_filtered["label"] = df_filtered["condition"].map(lambda c: LABELS.get(c, c))
    df_filtered["combo"] = df_filtered["protein"] + " / " + df_filtered["prior_level"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x="label", y="final_hv", hue="combo",
                order=[LABELS.get(c, c) for c in conds_present],
                palette=["#56B4E9", "#D55E00", "#009E73", "#CC79A7"][:len(df_filtered["combo"].unique())],
                ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("Final HV Distribution by Condition and Protein/Prior")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Protein / Prior", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_stagnant_batches(df, out_path):
    """Bar chart of stagnant batches (wasted experiment rounds)."""
    fig, axes = plt.subplots(1, len(df.groupby(["protein", "prior_level"])),
                             figsize=(5 * len(df.groupby(["protein", "prior_level"])), 5),
                             sharey=True)
    if len(df.groupby(["protein", "prior_level"])) == 1:
        axes = [axes]

    for ax, ((protein, prior), group) in zip(axes, df.groupby(["protein", "prior_level"])):
        conds = [c for c in ORDER if c in group["condition"].unique()]
        means = [group[group["condition"] == c]["stagnant_batches"].mean() for c in conds]
        stds = [group[group["condition"] == c]["stagnant_batches"].std() for c in conds]
        colors = [COLORS.get(c, "#333333") for c in conds]
        labels = [LABELS.get(c, c) for c in conds]

        ax.bar(range(len(conds)), means, yerr=stds, color=colors,
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{protein}\n{prior} prior", fontsize=10)
        ax.set_ylabel("Stagnant Batches (lower = better)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Stagnant Batches (zero HV improvement, lower = better)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="Path to raw results CSV")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same as results)")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Proteins: {df['protein'].unique()}")
    print(f"Priors: {df['prior_level'].unique()}")

    # Generate all figures
    fig_final_hv_grouped(df, out_dir / "fig1_final_hv.png")
    fig_exp_to_90pct(df, out_dir / "fig2_exp_to_90pct.png")
    fig_igd_comparison(df, out_dir / "fig3_igd.png")
    fig_hv_boxplot(df, out_dir / "fig4_hv_boxplot.png")
    fig_stagnant_batches(df, out_dir / "fig5_stagnant_batches.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
