"""
lengthscale_retrospective.py — Empirical validation of the lengthscale routing signal.

For every seed across all datasets: refit a GP on the init data and extract
the normalised lengthscale. Plot this against EGBO's advantage over random
on that seed to test whether lengthscale predicts which strategy wins.

If the correlation is strong, this:
  1. Validates lengthscale_norm as a routing signal
  2. Provides empirically calibrated thresholds (not guessed)
  3. Explains why the rule_router failed (thresholds were wrong)
  4. Motivates the n_init=15 experiment

Usage:
    python lengthscale_retrospective.py \\
        --results_dir results_phase1/ \\
        --data_dir data/
"""

import argparse
import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def fit_lengthscale(X_init: np.ndarray, y_init: np.ndarray,
                    bounds: np.ndarray) -> float:
    """Fit GP on init data and return normalised lengthscale."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_init)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_s, y_init)

    ls_scaled = float(gp.kernel_.length_scale)
    mean_scale = float(scaler.scale_.mean())
    mean_domain = float((bounds[:, 1] - bounds[:, 0]).mean())
    return float(np.clip(ls_scaled * mean_scale / (mean_domain + 1e-12), 0, 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_phase1")
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--n_init",      type=int, default=5)
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    data_dir    = pathlib.Path(args.data_dir)

    from oracle import NNOracle
    from shared_seed_experiment import generate_shared_inits

    DATASET_MAP = {
        "coatings":        "coatings",
        "pareto_20201218": "pareto_campaign 2020-12-18_17-38-40",
        "pareto_20210112": "pareto_campaign 2021-01-12_16-26-56",
        "hartmann3":       "hartmann3",
        "hartmann6":       "hartmann6",
    }
    STRUCTURED = {"coatings", "pareto_20210112", "hartmann3", "hartmann6"}

    rows = []

    for ds_label, ds_name in DATASET_MAP.items():
        cond_dir_egbo   = results_dir / ds_label / "egbo"
        cond_dir_random = results_dir / ds_label / "fixed_random"
        if not cond_dir_egbo.exists() or not cond_dir_random.exists():
            print(f"Skipping {ds_label}: missing results")
            continue

        print(f"Processing {ds_label}...")
        try:
            oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        except Exception as e:
            print(f"  Oracle load failed: {e}")
            continue

        bounds = oracle.bounds()
        gb     = oracle.global_best()
        N      = len(oracle._X_raw)
        budget = int(0.5 * N)

        # Load curves
        egbo_curves   = np.load(cond_dir_egbo   / "curves.npy")
        random_curves = np.load(cond_dir_random / "curves.npy")
        n_seeds = min(len(egbo_curves), len(random_curves))

        # Regenerate shared inits (same seed=42 as experiment)
        shared_inits = generate_shared_inits(oracle, n_seeds, args.n_init, rng_seed=42)

        for seed_idx in range(n_seeds):
            X_init, y_init = shared_inits[seed_idx]

            # Lengthscale from init data only
            try:
                ls_norm = fit_lengthscale(X_init, y_init, bounds)
            except Exception:
                ls_norm = float("nan")

            # Full-campaign AUC for egbo and random
            egbo_auc   = float(egbo_curves[seed_idx].mean()   / gb)
            random_auc = float(random_curves[seed_idx].mean() / gb)
            egbo_advantage = egbo_auc - random_auc

            # Init quality
            init_best_norm = float(y_init.max() / gb)

            rows.append({
                "dataset":         ds_label,
                "seed":            seed_idx,
                "ls_norm":         ls_norm,
                "egbo_auc":        egbo_auc,
                "random_auc":      random_auc,
                "egbo_advantage":  egbo_advantage,
                "init_best_norm":  init_best_norm,
                "structured":      ds_label in STRUCTURED,
            })

    df = pd.DataFrame(rows).dropna(subset=["ls_norm"])
    print(f"\nTotal seeds: {len(df)}")

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LENGTHSCALE vs EGBO ADVANTAGE")
    print("="*60)

    for ds in df.dataset.unique():
        sub = df[df.dataset == ds]
        r, p = stats.pearsonr(sub.ls_norm, sub.egbo_advantage)
        print(f"  {ds:<20} r={r:+.3f}  p={p:.3f}"
              + (" *" if p < 0.05 else ""))

    r_all, p_all = stats.pearsonr(df.ls_norm, df.egbo_advantage)
    print(f"\n  ALL DATASETS:       r={r_all:+.3f}  p={p_all:.4f}"
          + (" *" if p_all < 0.05 else ""))

    # Threshold analysis: what ls_norm threshold best separates egbo>random?
    print("\n" + "="*60)
    print("THRESHOLD CALIBRATION")
    print("="*60)
    thresholds = np.arange(0.05, 0.60, 0.05)
    print(f"  {'Threshold':>10}  {'Accuracy':>10}  {'N_above':>8}  {'N_below':>8}")
    best_acc, best_thresh = 0, 0.25
    for t in thresholds:
        above = df[df.ls_norm >= t]
        below = df[df.ls_norm <  t]
        if len(above) == 0 or len(below) == 0:
            continue
        # Accuracy: above threshold → predict egbo wins; below → predict random wins
        correct = (
            (above.egbo_advantage > 0).sum() +
            (below.egbo_advantage <= 0).sum()
        )
        acc = correct / len(df)
        marker = " ←" if acc > best_acc else ""
        print(f"  {t:>10.2f}  {acc:>10.3f}  {len(above):>8}  {len(below):>8}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"\n  Best threshold: ls_norm = {best_thresh:.2f}  "
          f"(accuracy = {best_acc:.3f})")
    print(f"  Rule-router used ls_rough=0.05, ls_smooth=0.25")
    print(f"  Empirically calibrated: {best_thresh:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter by dataset
    ax = axes[0]
    colours = {"coatings":"#2196F3", "pareto_20201218":"#F44336",
               "pareto_20210112":"#4CAF50", "hartmann3":"#FF9800",
               "hartmann6":"#9C27B0"}
    for ds in df.dataset.unique():
        sub = df[df.dataset == ds]
        ax.scatter(sub.ls_norm, sub.egbo_advantage,
                   label=ds, alpha=0.6, s=40,
                   color=colours.get(ds, "gray"))

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(best_thresh, color="red", lw=1.5, ls="--",
               label=f"threshold={best_thresh:.2f}")
    ax.set_xlabel("Lengthscale norm (from n_init observations)")
    ax.set_ylabel("EGBO advantage over Random (AUC)")
    ax.set_title("Lengthscale predicts EGBO advantage")
    ax.legend(fontsize=8)

    # Add regression line
    m, b = np.polyfit(df.ls_norm, df.egbo_advantage, 1)
    x_line = np.linspace(df.ls_norm.min(), df.ls_norm.max(), 100)
    ax.plot(x_line, m*x_line+b, "k-", lw=1.5, alpha=0.5)

    # Right: boxplot by regime (above/below threshold)
    ax2 = axes[1]
    above = df[df.ls_norm >= best_thresh].egbo_advantage
    below = df[df.ls_norm <  best_thresh].egbo_advantage
    ax2.boxplot([below, above],
                labels=[f"ls < {best_thresh:.2f}\n(route to LHS/random)",
                        f"ls ≥ {best_thresh:.2f}\n(route to EGBO)"])
    ax2.axhline(0, color="red", lw=1, ls="--")
    ax2.set_ylabel("EGBO advantage over Random (AUC)")
    ax2.set_title("EGBO advantage by routing decision")

    t_stat, p_box = stats.ttest_ind(above, below)
    ax2.set_xlabel(f"t-test: p={p_box:.3f}" + (" *" if p_box<0.05 else ""))

    plt.tight_layout()
    out_path = results_dir / "lengthscale_retrospective.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")

    # Save CSV
    csv_path = results_dir / "lengthscale_retrospective.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved data:   {csv_path}")

    print("\nConclusion:")
    if p_all < 0.05:
        print(f"  Lengthscale SIGNIFICANTLY predicts EGBO advantage (r={r_all:.3f}, p={p_all:.4f})")
        print(f"  Routing threshold ls_norm={best_thresh:.2f} achieves {best_acc:.0%} accuracy")
        print(f"  This validates the routing signal — but the signal needs n_init>5 to be reliable")
    else:
        print(f"  Lengthscale does NOT significantly predict EGBO advantage (r={r_all:.3f}, p={p_all:.4f})")
        print(f"  The routing signal itself is not informative from init data alone")
        print(f"  Implication: no simple signal predicts regime from n_init=5 observations")


if __name__ == "__main__":
    main()
