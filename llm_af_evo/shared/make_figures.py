"""
make_figures.py — llm_af_evo, build-up figures telling the L1 story:
  1. True-oracle-HV diagnostic: does EGBO-novelty beat pure exploitation
     with ground truth (mode A vs mode B)?
  2. Lambda sweep: does a GP-only, deployable exploration-credit fitness
     reproduce that separation without an oracle?
  3. Per-seed win-rate spread at lambda*, by condition — the noise-floor
     finding that motivated L1 over L2-first.
  4. The fitness-gaming discovery and its fix: evolution win-rate/fitness
     trajectory BEFORE (gameable GP-only proxy reused as training signal)
     vs AFTER (true-oracle training fitness).

Panels 1-3 are regenerated from data still on disk (logs/, via
diagnostic_true_oracle_hv.py and sweep_lambda.py's now-persisted JSON
summaries — run those two scripts first if diagnostic_summary.json /
sweep_lambda_summary.json don't exist yet). Panel 4's "before" series is
NOT regenerable — that run's history.json was overwritten by the corrected
rerun before this script existed. Those numbers are hardcoded below from
the actual printed output at the time (win_rate=1.0 from generation 0,
flat, final_population.json showing 8/8 identical degenerate
mu-free forms) — labeled explicitly as reconstructed-from-transcript, not
read from a live artifact. Preserve future run directories (e.g.
--out_dir evolution_runs/run_YYYYMMDD) before rerunning evolve_af.py if
you want this NOT to happen again.

Usage:
    python diagnostic_true_oracle_hv.py --logs_dir logs
    python sweep_lambda.py --logs_dir logs
    python make_figures.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent

# Hardcoded, transcript-sourced (see module docstring) — the pre-fix
# gamed evolution run's history, before evaluate_af was switched from the
# GP-only exploration-credit proxy to true-oracle HV as the training signal.
PRE_FIX_HISTORY = [
    {"generation": g, "best_win_rate": wr} for g, wr in
    [(0, 1.0), (1, 0.994), (2, 0.994), (3, 0.994), (4, 0.994),
     (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0), (9, 1.0), (10, 1.0)]
]


def panel_diagnostic(ax):
    path = HERE / "diagnostic_summary.json"
    if not path.exists():
        ax.text(0.5, 0.5, "run diagnostic_true_oracle_hv.py first",
                ha="center", va="center")
        return
    d = json.load(open(path))
    conds = sorted(d["by_condition"])
    rates = [d["by_condition"][c]["win_rate"] * 100 for c in conds]
    bars = ax.bar(conds + ["overall"], rates + [d["win_rate"] * 100],
                   color=["#4C72B0", "#4C72B0", "#55A868"])
    ax.axhline(50, color="gray", linestyle="--", linewidth=1, label="chance (50%)")
    ax.set_ylabel("EGBO-novelty win rate vs.\npure exploitation (true oracle HV)")
    ax.set_title("1. True-oracle-HV diagnostic")
    ax.set_ylim(0, 100)
    for b, r in zip(bars, rates + [d["win_rate"] * 100]):
        ax.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.1f}%", ha="center", fontsize=8)
    ax.legend(fontsize=8)


def panel_lambda_sweep(ax):
    path = HERE / "sweep_lambda_summary.json"
    if not path.exists():
        ax.text(0.5, 0.5, "run sweep_lambda.py first", ha="center", va="center")
        return
    d = json.load(open(path))
    lams = [max(row["lambda"], 1e-2) for row in d["sweep"]]  # avoid log(0)
    win_rates = [row["win_rate"] for row in d["sweep"]]
    corrs = [row["spearman"] for row in d["sweep"]]

    ax.plot(lams, win_rates, "o-", color="#4C72B0", label="win rate")
    ax.plot(lams, corrs, "s--", color="#C44E52", label="spearman r")
    ax.axhline(d.get("diagnostic_win_rate_ref", 120 / 180), color="gray",
               linestyle=":", linewidth=1, label="diagnostic win rate (0.667)")
    ax.axvline(d["best_lambda"], color="#55A868", linewidth=1,
               label=f"lambda*={d['best_lambda']}")
    ax.set_xscale("symlog")
    ax.set_xlabel("lambda")
    ax.set_title("2. Exploration-credit lambda sweep")
    ax.legend(fontsize=7, loc="lower right")


def panel_per_seed_variance(ax):
    path = HERE / "sweep_lambda_summary.json"
    if not path.exists():
        ax.text(0.5, 0.5, "run sweep_lambda.py first", ha="center", va="center")
        return
    d = json.load(open(path))
    per_seed = d["per_seed_win_rate_at_best_lambda"]
    by_cond = {}
    for key, rate in per_seed.items():
        cond = key.rsplit("_seed", 1)[0]
        by_cond.setdefault(cond, []).append(rate)

    labels = sorted(by_cond)
    data = [by_cond[c] for c in labels]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(f"per-seed win rate (lambda*={d['best_lambda']})")
    ax.set_title("3. Per-seed variance — not condition-explained")
    ax.tick_params(axis="x", rotation=20)


def panel_gaming_fix(ax):
    hist_path = HERE / "evolution_runs" / "run1" / "history.json"
    if not hist_path.exists():
        ax.text(0.5, 0.5, "run evolve_af.py first", ha="center", va="center")
        return
    post_raw = json.load(open(hist_path))
    # history.json was a flat list before evolve_af.py started also saving
    # n_llm_calls/n_llm_failures alongside it — support both shapes.
    post = post_raw["history"] if isinstance(post_raw, dict) else post_raw

    pre_x = [h["generation"] for h in PRE_FIX_HISTORY]
    pre_y = [h["best_win_rate"] for h in PRE_FIX_HISTORY]
    post_x = [h["generation"] for h in post]
    post_y = [h["best_win_rate"] for h in post]

    ax.plot(pre_x, pre_y, "o-", color="#C44E52",
            label="before fix (GP-only proxy as training fitness)")
    ax.plot(post_x, post_y, "o-", color="#4C72B0",
            label="after fix (true-oracle HV as training fitness)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("generation")
    ax.set_ylabel("best win rate vs. EGBO-novelty")
    ax.set_title("4. Fitness-gaming discovery and fix\n(pre-fix series reconstructed from run log, not a live artifact)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="center right")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panel_diagnostic(axes[0, 0])
    panel_lambda_sweep(axes[0, 1])
    panel_per_seed_variance(axes[1, 0])
    panel_gaming_fix(axes[1, 1])
    fig.suptitle("L1 build-up: from true-oracle diagnostic to a validated training fitness",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = HERE / "l1_buildup_figures.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
