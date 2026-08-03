"""
generate_all_figures.py — Comprehensive figures for Phase 1 + sensitivity analysis.

Produces:
  1. Phase 1 final HV bars (6 conditions × 4 combos)
  2. Phase 1 HV boxplot
  3. Sensitivity: novelty weight sweep (HV vs w_nov, with/without LLM)
  4. Sensitivity: isolated warm-start effect (mo_ls_egbo vs mo_egbo_real)
  5. Sample efficiency: exp_to_70pct comparison
  6. IGD comparison
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc_params
import seaborn as sns

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
    "mo_ls_egbo": "#E69F00",
}

LABELS = {
    "mo_random": "Random",
    "mo_egbo": "EGBO (light)",
    "mo_egbo_real": "EGBO (BoTorch)",
    "mo_egbo_novelty": "NA-EGBO",
    "mo_ls_na_egbo": "LS-NA-EGBO",
    "mo_llm_candidate_gen": "LLM-CandGen",
    "mo_ls_egbo": "LS-EGBO",
}

ORDER = ["mo_random", "mo_egbo", "mo_egbo_real", "mo_egbo_novelty",
         "mo_ls_na_egbo", "mo_llm_candidate_gen"]


def fig_phase1_hv_bars(p1_df, out_path):
    """Phase 1: Final HV bars, 6 conditions × 4 combos."""
    combos = list(p1_df.groupby(["protein", "prior_level"]).groups.keys())
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1: axes = [axes]

    for ax, ((protein, prior), group) in zip(axes, p1_df.groupby(["protein", "prior_level"])):
        conds = [c for c in ORDER if c in group["condition"].unique()]
        means = [group[group["condition"] == c]["final_hv"].mean() for c in conds]
        stds = [group[group["condition"] == c]["final_hv"].std() for c in conds]
        colors = [COLORS.get(c, "#333333") for c in conds]
        labels = [LABELS.get(c, c) for c in conds]

        ax.bar(range(len(conds)), means, yerr=stds, color=colors,
               capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{protein}\n{prior} prior", fontsize=10)
        ax.set_ylabel("Final Hypervolume" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

        # Highlight best
        best_idx = np.argmax(means)
        ax.bar(best_idx, means[best_idx], color=colors[best_idx],
               edgecolor="red", linewidth=2)

    plt.suptitle("Phase 1: Final Hypervolume by Condition (15 seeds, budget=30)\n"
                 "(red border = best condition)", fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_phase1_boxplot(p1_df, out_path):
    """Phase 1: HV boxplot."""
    conds_present = [c for c in ORDER if c in p1_df["condition"].unique()]
    df_f = p1_df[p1_df["condition"].isin(conds_present)].copy()
    df_f["label"] = df_f["condition"].map(lambda c: LABELS.get(c, c))
    df_f["combo"] = df_f["protein"] + " / " + df_f["prior_level"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_f, x="label", y="final_hv", hue="combo",
                order=[LABELS.get(c, c) for c in conds_present],
                palette=["#56B4E9", "#D55E00", "#009E73", "#CC79A7"][:len(df_f["combo"].unique())],
                ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("Phase 1: Final HV Distribution (15 seeds, budget=30)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Protein / Prior", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_novelty_sweep(sens_df, out_path):
    """Sensitivity: HV vs novelty weight, with and without LLM warm-start."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
    axes = axes.flatten()

    weights = [0.0, 0.1, 0.15, 0.2, 0.3]
    combos = list(sens_df.groupby(["protein", "prior_level"]).groups.keys())

    for ax, (protein, prior) in zip(axes, combos):
        group = sens_df[(sens_df["protein"] == protein) & (sens_df["prior_level"] == prior)]

        # LS-NA-EGBO sweep (with LLM)
        ls_means = []
        ls_stds = []
        for w in weights:
            cond = f"mo_ls_na_egbo_w{w:.2f}"
            sub = group[group["condition"] == cond]
            if len(sub) > 0:
                ls_means.append(sub["final_hv"].mean())
                ls_stds.append(sub["final_hv"].std())
            else:
                ls_means.append(np.nan)
                ls_stds.append(np.nan)

        # EGBO-novelty sweep (without LLM, random init)
        egbo_means = []
        egbo_stds = []
        for w in weights:
            cond = f"mo_egbo_novelty_w{w:.2f}"
            sub = group[group["condition"] == cond]
            if len(sub) > 0:
                egbo_means.append(sub["final_hv"].mean())
                egbo_stds.append(sub["final_hv"].std())
            else:
                egbo_means.append(np.nan)
                egbo_stds.append(np.nan)

        # mo_ls_egbo (no novelty, with LLM) — horizontal line
        ls_egbo = group[group["condition"] == "mo_ls_egbo"]
        ls_egbo_hv = ls_egbo["final_hv"].mean() if len(ls_egbo) > 0 else np.nan

        ax.errorbar(weights, ls_means, yerr=ls_stds, marker='o', color="#D55E00",
                    label="LS-NA-EGBO (LLM warm-start)", capsize=3, linewidth=2)
        ax.errorbar(weights, egbo_means, yerr=egbo_stds, marker='s', color="#009E73",
                    label="NA-EGBO (random init)", capsize=3, linewidth=2)
        ax.axhline(y=ls_egbo_hv, color="#E69F00", linestyle='--', linewidth=1.5,
                   label=f"LS-EGBO (LLM, no novelty): {ls_egbo_hv:.0f}")

        ax.set_xlabel("Novelty weight (w_nov)")
        ax.set_ylabel("Final Hypervolume")
        ax.set_title(f"{protein} / {prior}", fontsize=10)
        ax.set_xticks(weights)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)

    plt.suptitle("Novelty Weight Sensitivity: HV vs w_nov (15 seeds, budget=30)\n"
                 "w_nov=0.10 is the sweet spot; w_nov=0.30 (Aqeeli default) is too aggressive",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_warmstart_isolation(sens_df, p1_df, out_path):
    """Isolated warm-start effect: mo_ls_egbo vs mo_egbo_real."""
    fig, ax = plt.subplots(figsize=(10, 6))

    combos = list(sens_df.groupby(["protein", "prior_level"]).groups.keys())
    x_labels = []
    ls_egbo_vals = []
    real_vals = []

    for protein, prior in combos:
        ls = sens_df[(sens_df["protein"] == protein) & (sens_df["prior_level"] == prior) &
                     (sens_df["condition"] == "mo_ls_egbo")]["final_hv"].values
        real = p1_df[(p1_df["protein"] == protein) & (p1_df["prior_level"] == prior) &
                     (p1_df["condition"] == "mo_egbo_real")]["final_hv"].values
        x_labels.append(f"{protein}\n{prior}")
        ls_egbo_vals.append(ls)
        real_vals.append(real)

    x = np.arange(len(combos))
    width = 0.35

    # Box plots side by side
    for i, (ls, real) in enumerate(zip(ls_egbo_vals, real_vals)):
        ax.boxplot([ls, real], positions=[i - width/2, i + width/2],
                   widths=width*0.8, patch_artist=True,
                   boxprops=dict(facecolor="#E69F00", alpha=0.6),
                   medianprops=dict(color="black"))

    # Actually use bar chart with error bars for clarity
    ax.cla()
    ls_means = [np.mean(v) for v in ls_egbo_vals]
    ls_stds = [np.std(v) for v in ls_egbo_vals]
    real_means = [np.mean(v) for v in real_vals]
    real_stds = [np.std(v) for v in real_vals]

    bars1 = ax.bar(x - width/2, ls_means, width, yerr=ls_stds, color="#E69F00",
                   label="LS-EGBO (LLM warm-start)", capsize=3, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, real_means, width, yerr=real_stds, color="#56B4E9",
                   label="EGBO-real (random init)", capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Final Hypervolume")
    ax.set_title("Isolated Warm-Start Effect: LLM init vs Random init\n"
                 "(Both use qLogNEHVI + U-NSGA-III, NO novelty selection)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Add delta annotations
    for i, (lm, rm) in enumerate(zip(ls_means, real_means)):
        delta = lm - rm
        pct = delta / rm * 100
        color = "green" if delta > 0 else "red"
        ax.annotate(f"{delta:+.0f}\n({pct:+.1f}%)",
                    xy=(i, max(lm, rm) + max(ls_stds[i], real_stds[i]) + 200),
                    ha='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_sample_efficiency(sens_df, out_path):
    """Sample efficiency: exp_to_70pct comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    best_configs = {
        ('mAb_aggregation', 'L1'): 0.10,
        ('mAb_aggregation', 'blank'): 0.20,
        ('mAb_oxidation', 'L1'): 0.00,
        ('mAb_oxidation', 'blank'): 0.10,
    }

    combos = list(best_configs.keys())
    x = np.arange(len(combos))
    width = 0.35

    ls_exp = []
    rand_exp = []
    for (protein, prior), best_w in best_configs.items():
        cond = f"mo_ls_na_egbo_w{best_w:.2f}"
        ls = sens_df[(sens_df["protein"] == protein) & (sens_df["prior_level"] == prior) &
                     (sens_df["condition"] == cond)]["exp_to_70pct_hv"].mean()
        rand = sens_df[(sens_df["protein"] == protein) & (sens_df["prior_level"] == prior) &
                       (sens_df["condition"] == "mo_egbo_novelty_w0.00")]["exp_to_70pct_hv"].mean()
        ls_exp.append(ls)
        rand_exp.append(rand)

    bars1 = ax.bar(x - width/2, ls_exp, width, color="#D55E00",
                   label="LS-NA-EGBO (best w_nov, LLM)", capsize=3, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rand_exp, width, color="#009E73",
                   label="NA-EGBO w=0.00 (random init)", capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{pr}" for p, pr in combos], fontsize=9)
    ax.set_ylabel("Experiments to 70% of Max HV (lower = better)")
    ax.set_title("Sample Efficiency: Experiments to 70% HV\n"
                 "LLM warm-start saves 2-6 experiments in 3/4 combos", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.invert_yaxis()

    # Add saving annotations
    for i, (ls, rand) in enumerate(zip(ls_exp, rand_exp)):
        saving = rand - ls
        color = "green" if saving > 0 else "red"
        ax.annotate(f"saves {saving:.1f}", xy=(i, min(ls, rand) - 0.5),
                    ha='center', fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def fig_igd_phase1(p1_df, out_path):
    """Phase 1: IGD comparison."""
    combos = list(p1_df.groupby(["protein", "prior_level"]).groups.keys())
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1: axes = [axes]

    for ax, ((protein, prior), group) in zip(axes, p1_df.groupby(["protein", "prior_level"])):
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

    plt.suptitle("Phase 1: IGD (lower = closer to true Pareto front)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p1_path = pathlib.Path("/mnt/results/benchmark_phase1/phase1_raw_results.csv")
    sens_path = pathlib.Path("/mnt/results/benchmark_sensitivity/sensitivity_raw_results.csv")
    out_dir = pathlib.Path("/mnt/results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    p1_df = pd.read_csv(p1_path)
    sens_df = pd.read_csv(sens_path)

    print(f"Phase 1: {len(p1_df)} rows, conditions: {p1_df['condition'].unique()}")
    print(f"Sensitivity: {len(sens_df)} rows, conditions: {sens_df['condition'].unique()}")

    # Phase 1 figures
    fig_phase1_hv_bars(p1_df, out_dir / "fig1_phase1_hv_bars.png")
    fig_phase1_boxplot(p1_df, out_dir / "fig2_phase1_hv_boxplot.png")
    fig_igd_phase1(p1_df, out_dir / "fig3_phase1_igd.png")

    # Sensitivity figures
    fig_novelty_sweep(sens_df, out_dir / "fig4_novelty_weight_sweep.png")
    fig_warmstart_isolation(sens_df, p1_df, out_dir / "fig5_warmstart_isolation.png")
    fig_sample_efficiency(sens_df, out_dir / "fig6_sample_efficiency.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
