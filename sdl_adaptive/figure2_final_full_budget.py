"""Figure 2d — final normalised best at FULL budget (not step 40), as a bar
chart, using the same range-normalisation as figure2_short_horizon.py's
kind='final' (val - global_min) / (global_best - global_min) computed
directly off curves.npy — NOT metrics_summary.csv's `final_best_normalised`
column, which divides by global_best alone (un-range-normalised) and would
not be apples-to-apples with the step-40 figure2c.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='ticks', font='DejaVu Sans')

RESULTS_DIR = pathlib.Path("results_combined")
OUT_DIR = pathlib.Path("figures_combined")
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ['coatings', 'pareto_20201218', 'pareto_20201223', 'pareto_20210104', 'pareto_20210112']
DS_SHORT = {
    'coatings':        'Coatings\n(7D)',
    'pareto_20201218': 'Pareto\n12-18',
    'pareto_20201223': 'Pareto\n12-23',
    'pareto_20210104': 'Pareto\n01-04',
    'pareto_20210112': 'Pareto\n01-12',
}
COND_LABELS = {
    'fixed_random': 'Random', 'fixed_lhs': 'LHS', 'fixed_ucb_low': 'UCB β=0.2',
    'fixed_ucb_high': 'UCB β=400', 'fixed_ei': 'EI (fixed)', 'ada_original': 'ADA orig.',
    'mock_approach_a': 'Mock-A', 'mock_approach_b': 'Mock-B', 'mock_approach_c': 'Mock-C',
    'approach_a': 'LLM-A', 'approach_b': 'LLM-B', 'approach_c': 'LLM-C',
    'approach_d': 'LLM-D', 'egbo': 'EGBO', 'novelty_egbo': 'Novelty-EGBO',
}
COND_ORDER = ['fixed_random', 'fixed_lhs', 'fixed_ucb_low', 'fixed_ucb_high', 'fixed_ei',
              'ada_original', 'mock_approach_a', 'mock_approach_b', 'mock_approach_c',
              'approach_a', 'approach_b', 'approach_c', 'approach_d', 'egbo', 'novelty_egbo']
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442',
           '#000000', '#999999', '#332288', '#117733', '#AA4499', '#88CCEE', '#44AA99', '#661100']
COLORS = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(COND_ORDER)}

df_summary = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")

fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

for ax, ds in zip(axes, DATASETS):
    ds_dir = RESULTS_DIR / ds
    row = df_summary[df_summary.dataset == ds].iloc[0]
    gb, gmin = float(row.global_best), float(row.global_min)
    y_range = gb - gmin

    conds_present = [c for c in COND_ORDER if (ds_dir / c / "curves.npy").exists()]
    means, stds = [], []
    for c in conds_present:
        curves = np.load(ds_dir / c / "curves.npy")  # (n_seeds, full budget)
        final = curves[:, -1]  # last step = full budget, whatever it is for this dataset
        normed = np.clip((final - gmin) / y_range, 0, 1)
        means.append(normed.mean())
        stds.append(normed.std())

    x = np.arange(len(conds_present))
    ax.bar(x, means, yerr=stds, color=[COLORS[c] for c in conds_present],
           capsize=3, error_kw={'linewidth': 1.2}, width=0.7, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds_present],
                        rotation=45, ha='right', fontsize=7)
    ax.set_title(DS_SHORT[ds], fontsize=9)
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)

axes[0].set_ylabel('Final normalised best @ full budget (mean ± SD)', fontsize=9)
fig.suptitle('Figure 2d — Best-found-so-far at full budget per condition and dataset',
             fontsize=11, y=1.01)
plt.tight_layout()
out = OUT_DIR / 'figure2_final_full_budget.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")
