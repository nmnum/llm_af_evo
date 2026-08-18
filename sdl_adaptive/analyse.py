"""
analyse.py — Generate all figures from experiment results.

Usage
-----
python analyse.py --results_dir results/ --out_dir figures/

Figures produced
----------------
figure1_efficiency.png      — Running-best curves (mean ± 1 SD, 20 seeds)
figure2_auc_bars.png        — AUC bar chart per condition and dataset
figure3_decision_quality.png— AUC per condition (bar chart with sw= annotations)
figure4_strategy_heatmap.png— Strategy usage heatmap for mock_b and mock_c
"""

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.rcParams['font.family'] = ['Helvetica', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='ticks', font='Helvetica')

# ── Shared display config ─────────────────────────────────────────────────────
COND_LABELS = {
    'fixed_random':    'Random',
    'fixed_lhs':       'LHS',
    'fixed_ucb_low':   'UCB β=0.2',
    'fixed_ucb_high':  'UCB β=400',
    'fixed_ei':        'EI (fixed)',
    'ada_original':    'ADA orig.',
    'mock_approach_a': 'Mock-A',
    'mock_approach_b': 'Mock-B',
    'mock_approach_c': 'Mock-C',
    'approach_a':      'LLM-A',
    'approach_b':      'LLM-B',
    'approach_c':      'LLM-C',
    'approach_c_evo':  'LLM-C-evo',
    'approach_d':      'LLM-D',
    'egbo':            'EGBO',
    'novelty_egbo':    'Novelty-EGBO',
}

# Okabe-Ito colorblind-safe palette (extended to 12)
PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9',
    '#D55E00', '#F0E442', '#000000', '#999999',
    '#332288', '#117733', '#AA4499',
]

DATASETS = ['coatings', 'pareto_20201218', 'pareto_20201223', 'pareto_20210104', 'pareto_20210112']
DS_LABELS = {
    'coatings':        'Coatings 2022\n(7D, single-obj)',
    'pareto_20201218': 'Pareto 2020-12-18\n(4D, multi-obj)',
    'pareto_20201223': 'Pareto 2020-12-23\n(4D, multi-obj)',
    'pareto_20210104': 'Pareto 2021-01-04\n(4D, multi-obj)',
    'pareto_20210112': 'Pareto 2021-01-12\n(4D, multi-obj)',
}
DS_SHORT = {
    'coatings':        'Coatings\n(7D)',
    'pareto_20201218': 'Pareto\n12-18',
    'pareto_20201223': 'Pareto\n12-23',
    'pareto_20210104': 'Pareto\n01-04',
    'pareto_20210112': 'Pareto\n01-12',
}

LINESTYLES = {
    'fixed_random': '--', 'fixed_lhs': ':', 'fixed_ucb_low': '-.',
    'fixed_ucb_high': '-.', 'fixed_ei': '-', 'ada_original': (0, (3, 1, 1, 1)),
    'mock_approach_a': '-', 'mock_approach_b': '-', 'mock_approach_c': '-',
    'approach_a': '-', 'approach_b': '-', 'approach_c': '-',
    'approach_c_evo': '-', 'approach_d': '-',
    'egbo': '--', 'novelty_egbo': '--',
}


def _get_colors_and_order(df):
    """Build color/order dicts from conditions present in df."""
    present = list(df['condition'].unique())
    # Preserve canonical order, append any extras
    canonical = list(COND_LABELS.keys())
    order = [c for c in canonical if c in present] + [c for c in present if c not in canonical]
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(order)}
    return order, colors


# ── Figure 1: Efficiency curves ───────────────────────────────────────────────
def figure1_efficiency(df, results_dir, out_dir):
    cond_order, colors = _get_colors_and_order(df)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        ds_dir = results_dir / ds
        oracle_min = df[df['dataset'] == ds]['global_min'].iloc[0]
        oracle_max = df[df['dataset'] == ds]['global_best'].iloc[0]
        y_range = oracle_max - oracle_min

        for cond in cond_order:
            curves_path = ds_dir / cond / 'curves.npy'
            if not curves_path.exists():
                continue
            curves = np.load(curves_path)
            curves_norm = np.clip((curves - oracle_min) / y_range, 0, 1)
            mean_c = curves_norm.mean(axis=0)
            std_c = curves_norm.std(axis=0)
            steps = np.arange(len(mean_c))
            ls = LINESTYLES.get(cond, '-')
            ax.plot(steps, mean_c, color=colors[cond], linestyle=ls,
                    linewidth=1.6, label=COND_LABELS.get(cond, cond), alpha=0.9)
            ax.fill_between(steps, mean_c - std_c, mean_c + std_c,
                            color=colors[cond], alpha=0.10)

        ax.set_title(DS_LABELS[ds], fontsize=9)
        ax.set_xlabel('Experiment step', fontsize=8)
        ax.set_ylim(0, 1.05)
        sns.despine(ax=ax)

    axes[0].set_ylabel('Normalised running best', fontsize=9)
    handles = [mpatches.Patch(color=colors[c], label=COND_LABELS.get(c, c))
               for c in cond_order]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle('Figure 1 — Efficiency: Running-best curves (mean ± 1 SD, 20 seeds)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = out_dir / 'figure1_efficiency.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 2: AUC bar chart ───────────────────────────────────────────────────
def figure2_auc_bars(df, out_dir):
    cond_order, colors = _get_colors_and_order(df)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        sub = df[df['dataset'] == ds]
        conds = [c for c in cond_order if c in sub['condition'].values]
        means = [sub[sub['condition'] == c]['auc_best'].mean() for c in conds]
        stds  = [sub[sub['condition'] == c]['auc_best'].std()  for c in conds]
        x = np.arange(len(conds))
        ax.bar(x, means, yerr=stds, color=[colors[c] for c in conds],
               capsize=3, error_kw={'linewidth': 1.2}, width=0.7, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds],
                           rotation=45, ha='right', fontsize=7)
        ax.set_title(DS_SHORT[ds], fontsize=9)
        ax.set_ylim(0, 1.05)
        sns.despine(ax=ax)

    axes[0].set_ylabel('AUC (normalised, mean ± SD)', fontsize=9)
    fig.suptitle('Figure 2 — AUC per condition and dataset (20 seeds)', fontsize=11, y=1.01)
    plt.tight_layout()
    out = out_dir / 'figure2_auc_bars.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 3: Decision quality (bar chart with sw= annotations) ───────────────
def figure3_decision_quality(df, out_dir):
    cond_order, colors = _get_colors_and_order(df)
    agg = (df.groupby(['condition', 'dataset'])
             .agg(auc_mean=('auc_best', 'mean'),
                  auc_std=('auc_best', 'std'),
                  sw_mean=('switch_frequency', 'mean'))
             .reset_index())

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        sub = agg[agg['dataset'] == ds].set_index('condition')
        for i, cond in enumerate(cond_order):
            if cond not in sub.index:
                continue
            row = sub.loc[cond]
            ax.bar(i, row['auc_mean'], color=colors[cond], width=0.7, alpha=0.85, zorder=2)
            ax.errorbar(i, row['auc_mean'], yerr=row['auc_std'],
                        fmt='none', color='#333333', capsize=3, linewidth=1.2, zorder=3)
            if row['sw_mean'] > 0.001:
                ax.text(i, row['auc_mean'] + row['auc_std'] + 0.015,
                        f"sw={row['sw_mean']:.2f}",
                        ha='center', va='bottom', fontsize=6.5, color='#333333')

        ax.set_xticks([])
        ax.set_xlim(-0.6, len(cond_order) - 0.4)
        ax.set_ylim(0.0, 1.15)
        ax.set_title(DS_SHORT[ds], fontsize=9, pad=6)
        ax.axhline(0.5, color='grey', lw=0.7, ls='--', zorder=1)
        sns.despine(ax=ax)

    axes[0].set_ylabel('AUC (normalised, mean ± SD, 20 seeds)', fontsize=9)
    handles = [mpatches.Patch(color=colors[c], label=COND_LABELS.get(c, c))
               for c in cond_order]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.suptitle(
        'Figure 3 — Decision quality: AUC per condition and dataset\n'
        '(sw= annotation shows mean switch frequency for adaptive controllers)',
        fontsize=10, y=1.02)
    plt.tight_layout()
    out = out_dir / 'figure3_decision_quality.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── Figure 4: Strategy usage heatmap ─────────────────────────────────────────
def figure4_strategy_heatmap(results_dir, out_dir):
    STRATEGY_NAMES = ['ucb', 'ei', 'pi', 'thompson', 'random', 'lhs', '_custom']
    STRATEGY_DISPLAY = {'ucb': 'UCB', 'ei': 'EI', 'pi': 'PI', 'thompson': 'Thompson',
                        'random': 'Random', 'lhs': 'LHS', '_custom': 'Custom (code)'}
    N_WINDOWS = 5

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))

    for col, ds in enumerate(DATASETS):
        for row, (cond, cond_label) in enumerate([
            ('mock_approach_b', 'Mock-B (strategy switch)'),
            ('mock_approach_c', 'Mock-C (code rewrite)'),
        ]):
            ax = axes[row, col]
            logs_path = results_dir / ds / cond / 'switch_logs.json'
            curves_path = results_dir / ds / cond / 'curves.npy'

            if not logs_path.exists():
                ax.set_visible(False)
                continue

            with open(logs_path) as f:
                logs = json.load(f)
            curves = np.load(curves_path)
            budget = curves.shape[1]
            window_size = max(budget // N_WINDOWS, 1)

            counts = np.zeros((len(STRATEGY_NAMES), N_WINDOWS))
            for seed_log in logs:
                for step_idx, (t, strategy, _) in enumerate(seed_log['decisions']):
                    window = min(step_idx // window_size, N_WINDOWS - 1)
                    if strategy in STRATEGY_NAMES:
                        counts[STRATEGY_NAMES.index(strategy), window] += 1

            col_sums = counts.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            hm = counts / col_sums

            nonzero = hm.sum(axis=1) > 0
            hm_plot = hm[nonzero]
            row_labels = [STRATEGY_DISPLAY.get(STRATEGY_NAMES[i], STRATEGY_NAMES[i])
                          for i in range(len(STRATEGY_NAMES)) if nonzero[i]]
            win_labels = [f'{int(100*w/N_WINDOWS)}–{int(100*(w+1)/N_WINDOWS)}%'
                          for w in range(N_WINDOWS)]

            sns.heatmap(hm_plot, ax=ax, cmap='Blues', vmin=0, vmax=1,
                        xticklabels=win_labels, yticklabels=row_labels,
                        annot=True, fmt='.2f', annot_kws={'size': 7},
                        cbar=(col == 4), linewidths=0.3)
            ax.set_xticklabels(win_labels, fontsize=7, rotation=30)
            ax.set_yticklabels(row_labels, fontsize=7, rotation=0)
            if col == 0:
                ax.set_ylabel(cond_label, fontsize=8, fontweight='bold')
            if row == 0:
                ax.set_title(DS_SHORT[ds], fontsize=8)

    fig.suptitle(
        'Figure 4 — Strategy usage heatmap: fraction of steps per strategy per campaign phase\n'
        '(averaged across 20 seeds)', fontsize=10, y=1.01)
    plt.tight_layout()
    out = out_dir / 'figure4_strategy_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SDL experiment figures")
    parser.add_argument("--results_dir", default="results", help="Experiment results directory")
    parser.add_argument("--out_dir", default="figures", help="Output directory for figures")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "metrics_summary.csv")
    print(f"Loaded metrics_summary.csv: {len(df)} rows\n")

    print("Generating figures...")
    figure1_efficiency(df, results_dir, out_dir)
    figure2_auc_bars(df, out_dir)
    figure3_decision_quality(df, out_dir)
    figure4_strategy_heatmap(results_dir, out_dir)
    print("\nDone.")
