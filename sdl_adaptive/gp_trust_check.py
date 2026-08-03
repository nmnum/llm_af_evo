"""
gp_trust_check.py — Proactive GP trustworthiness diagnostics.

Three diagnostics computed from initial observations only (n=5–20).
Designed to fire at campaign start, before committing experiments.

DIAGNOSTIC 1: LOO calibration
  Fit GP on n-1 points, predict left-out point.
  Compute standardised residual z_i = (y_pred_i - y_true_i) / sigma_pred_i.
  If residuals ~ N(0,1): GP is calibrated (trustworthy).
  If |z| >> 1 systematically: GP is overconfident (wrong lengthscale or prior).
  If |z| << 1 systematically: GP is underconfident (overly cautious, but safe).

DIAGNOSTIC 2: Lengthscale / mean nearest-neighbour distance ratio
  ls_nn_ratio = ls_fitted / mean_nn_distance (both in normalised input space).
  If ls_nn_ratio < 1.0: GP claims structure at finer scale than data spacing.
    → GP is fitting noise. Not trustworthy.
  If ls_nn_ratio > 1.0: GP lengthscale is consistent with data coverage.
    → GP may be trustworthy (necessary but not sufficient).
  If ls_nn_ratio >> 5.0: GP assumes very smooth landscape.
    → Check LOO — if calibrated, trust it. If not, prior is too strong.

DIAGNOSTIC 3: Posterior uncertainty reduction vs prior
  reduction = 1 - (mean posterior sigma / prior sigma)
  If reduction ~ 0: GP hasn't learned from data (prior dominates).
    → Either data is uninformative OR prior is too strong.
  If reduction > 0.3: GP has updated substantially from data.
    → GP is responding to observations (good sign).
  Danger zone: low posterior sigma + high LOO error = confident but wrong.

COMPOSITE TRUST SCORE: weighted combination → [0, 1]
  0.0 = do not trust GP (use LHS/random)
  0.5 = uncertain (use with caution, monitor)
  1.0 = trust GP (use EGBO/UCB)

Usage:
    from gp_trust_check import gp_trust_check, interpret_trust

    result = gp_trust_check(X_init, y_init, bounds)
    print(interpret_trust(result))

    # With meta-learned prior (PACOH):
    result = gp_trust_check(X_init, y_init, bounds, prior_lengthscale=0.4)
"""

import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrustResult:
    """Full diagnostic output from gp_trust_check()."""
    # Composite
    trust_score: float          # [0,1] — overall trustworthiness
    trust_flag: str             # "trust" / "caution" / "distrust"

    # Diagnostic 1: LOO calibration
    loo_mean_abs_z: float       # mean |standardised residual|. ~0.8 = ideal
    loo_std_z: float            # std of residuals. ~1.0 = ideal
    loo_calibration: str        # "calibrated" / "overconfident" / "underconfident"
    loo_residuals: np.ndarray   # raw residuals for inspection

    # Diagnostic 2: ls/nn_distance ratio
    ls_fitted: float            # fitted lengthscale (normalised space)
    mean_nn_distance: float     # mean nearest-neighbour distance
    ls_nn_ratio: float          # ls_fitted / mean_nn_distance
    ls_flag: str                # "ok" / "overfit" / "smooth"

    # Diagnostic 3: uncertainty reduction
    prior_sigma: float          # prior predictive std (before data)
    posterior_sigma: float      # posterior predictive std (after data)
    uncertainty_reduction: float  # 1 - posterior/prior. >0.3 = GP learned
    unc_flag: str               # "learned" / "uninformative" / "overconfident"

    # Metadata
    n_obs: int
    n_dims: int
    notes: list = field(default_factory=list)


def gp_trust_check(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    bounds: np.ndarray,
    prior_lengthscale: Optional[float] = None,
    n_test_points: int = 200,
    verbose: bool = False,
) -> TrustResult:
    """
    Run the three GP trustworthiness diagnostics on initial observations.

    Parameters
    ----------
    X_obs            : (n, d) observed inputs
    y_obs            : (n,)   observed outputs
    bounds           : (d, 2) input bounds [lo, hi] per dimension
    prior_lengthscale: if provided (e.g. from PACOH meta-learning),
                       use this as the initial lengthscale for the GP.
                       If None, fit from scratch (default mode).
    n_test_points    : random test points for posterior sigma estimate
    verbose          : print diagnostic summary

    Returns
    -------
    TrustResult dataclass with all diagnostic values and composite trust score.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")

    n, d = X_obs.shape
    notes = []

    if n < 3:
        notes.append("Too few observations for reliable diagnostics (need n≥3)")
        return TrustResult(
            trust_score=0.5, trust_flag="caution",
            loo_mean_abs_z=float('nan'), loo_std_z=float('nan'),
            loo_calibration="insufficient_data",
            loo_residuals=np.array([]),
            ls_fitted=float('nan'), mean_nn_distance=float('nan'),
            ls_nn_ratio=float('nan'), ls_flag="insufficient_data",
            prior_sigma=float('nan'), posterior_sigma=float('nan'),
            uncertainty_reduction=float('nan'), unc_flag="insufficient_data",
            n_obs=n, n_dims=d, notes=notes,
        )

    # Normalise inputs to [0,1] per dimension
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo + 1e-12
    X_norm = (X_obs - lo) / span

    # Standardise outputs
    y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
    y_scaled = (y_obs - y_mean) / y_std

    # ── Build GP kernel ───────────────────────────────────────────────────────
    if prior_lengthscale is not None:
        # Use meta-learned prior lengthscale as starting point
        # Tighter bounds force it to stay near the prior
        kernel = Matern(
            nu=2.5,
            length_scale=prior_lengthscale,
            length_scale_bounds=(prior_lengthscale * 0.5,
                                  prior_lengthscale * 2.0),
        )
        notes.append(f"Using prior lengthscale: {prior_lengthscale:.3f}")
    else:
        kernel = Matern(
            nu=2.5,
            length_scale=0.5,
            length_scale_bounds=(1e-3, 10.0),
        )

    def _fit_gp(X, y):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,  # already standardised
            n_restarts_optimizer=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y)
        return gp

    # ── Full GP fit (for lengthscale and posterior sigma) ─────────────────────
    gp_full = _fit_gp(X_norm, y_scaled)
    ls_fitted = float(gp_full.kernel_.length_scale)

    # Prior sigma: GP prediction at a point far from all observations
    # Use the kernel's prior: k(x,x) = amplitude (≈1 for normalised outputs)
    prior_sigma = float(np.sqrt(gp_full.kernel_.k1.constant_value
                                if hasattr(gp_full.kernel_, 'k1') else 1.0))

    # Posterior sigma: mean over random test points
    rng = np.random.default_rng(42)
    test_pts = rng.random((n_test_points, d))
    _, sigma_post = gp_full.predict(test_pts, return_std=True)
    posterior_sigma = float(sigma_post.mean())

    # Prior sigma from kernel (variance before conditioning)
    # For Matern with default amplitude=1, prior std=1 in standardised space
    prior_sigma = 1.0  # standardised outputs → prior std ≈ 1

    uncertainty_reduction = float(1.0 - posterior_sigma / prior_sigma)
    uncertainty_reduction = float(np.clip(uncertainty_reduction, -1, 1))

    # ── Diagnostic 2: ls / mean_nn_distance ──────────────────────────────────
    # Mean nearest-neighbour distance in normalised input space
    dists = np.sqrt(((X_norm[:, None] - X_norm[None, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    nn_distances = dists.min(axis=1)
    mean_nn_distance = float(nn_distances.mean())

    ls_nn_ratio = ls_fitted / (mean_nn_distance + 1e-12)

    obs_per_dim = n / d
    ls_lower_bound = 1e-3  # sklearn default lower bound

    # When n/d < 2, lengthscale is not identifiable — skip ls diagnostic
    if obs_per_dim < 2.0:
        ls_flag = "sparse"
        notes.append(
            f"n/d={obs_per_dim:.1f} < 2: too sparse for reliable lengthscale "
            f"estimation (d={d}, n={n}) — ls diagnostic skipped"
        )
    elif abs(ls_fitted - ls_lower_bound) < 1e-4:
        # Lengthscale hit the lower bound → optimisation failed, not overfit
        ls_flag = "sparse"
        notes.append(
            f"ls={ls_fitted:.4f} hit lower bound: GP cannot estimate "
            f"lengthscale with n={n} in d={d} — ls diagnostic unreliable"
        )
    elif ls_nn_ratio < 0.8:
        ls_flag = "overfit"
        notes.append(
            f"ls/nn={ls_nn_ratio:.2f} < 0.8: GP fitting noise "
            f"(ls={ls_fitted:.3f} < nn={mean_nn_distance:.3f})"
        )
    elif ls_nn_ratio > 8.0:
        ls_flag = "smooth"
        notes.append(
            f"ls/nn={ls_nn_ratio:.2f} > 8: GP assumes very smooth landscape"
        )
    else:
        ls_flag = "ok"

    # ── Diagnostic 1: LOO calibration ────────────────────────────────────────
    loo_residuals = np.zeros(n)
    loo_z = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X_norm[mask], y_scaled[mask]
        X_test,  y_test  = X_norm[[i]], y_scaled[i]

        if len(X_train) < 2:
            loo_z[i] = 0.0
            continue

        gp_loo = _fit_gp(X_train, y_train)
        mu_pred, sigma_pred = gp_loo.predict(X_test, return_std=True)
        sigma_pred = max(float(sigma_pred[0]), 1e-6)
        residual = float(mu_pred[0]) - float(y_test)
        loo_residuals[i] = residual
        loo_z[i] = residual / sigma_pred

    loo_mean_abs_z = float(np.abs(loo_z).mean())
    loo_std_z = float(loo_z.std())

    # Ideal: |z| ~ 0.8 (half-normal of N(0,1)), std(z) ~ 1.0
    # Overconfident: |z| >> 1 (predictions too narrow)
    # Underconfident: |z| << 0.5 (predictions too wide)
    if loo_mean_abs_z > 1.8:
        loo_calibration = "overconfident"
        notes.append(
            f"LOO mean|z|={loo_mean_abs_z:.2f} > 1.8: GP overconfident "
            f"(predictions too narrow for actual errors)"
        )
    elif loo_mean_abs_z < 0.3:
        loo_calibration = "underconfident"
        notes.append(
            f"LOO mean|z|={loo_mean_abs_z:.2f} < 0.3: GP underconfident "
            f"(very conservative — safe but may be uninformative)"
        )
    else:
        loo_calibration = "calibrated"

    # ── Diagnostic 3: uncertainty reduction ──────────────────────────────────
    if uncertainty_reduction < 0.1:
        unc_flag = "uninformative"
        notes.append(
            f"Posterior σ ≈ prior σ (reduction={uncertainty_reduction:.2f}): "
            f"GP hasn't learned from data"
        )
    elif uncertainty_reduction > 0.3 and loo_mean_abs_z > 1.8:
        unc_flag = "overconfident"
        notes.append(
            "DANGER: GP is confident (low σ) but LOO errors are large. "
            "Transferred prior may be misleading."
        )
    else:
        unc_flag = "learned"

    # ── Composite trust score ─────────────────────────────────────────────────
    # Each diagnostic contributes to score [0,1]
    # LOO calibration (most important): weight 0.5
    # ls/nn ratio: weight 0.3
    # uncertainty reduction: weight 0.2

    # LOO score: 1.0 if calibrated, 0.0 if overconfident, 0.5 if underconfident
    if loo_calibration == "calibrated":
        loo_score = 1.0
    elif loo_calibration == "overconfident":
        # Graded: slightly over is okay, wildly over is bad
        loo_score = max(0.0, 1.0 - (loo_mean_abs_z - 1.8) / 2.0)
    else:  # underconfident
        loo_score = 0.6  # conservative but not dangerous

    # ls/nn score: 0 if overfit, 0.5 if very smooth, 1 if ok/sparse
    if ls_flag == "overfit":
        ls_score = 0.0
    elif ls_flag == "smooth":
        ls_score = 0.5
    elif ls_flag == "sparse":
        ls_score = 0.5   # neutral — not informative either way
    else:
        ls_score = 1.0

    # Uncertainty reduction score
    if unc_flag == "overconfident":
        unc_score = 0.0  # dangerous
    elif unc_flag == "uninformative":
        unc_score = 0.5  # safe but useless
    else:
        unc_score = 1.0

    # When ls is sparse (unreliable), upweight LOO and downweight ls
    if ls_flag == "sparse":
        trust_score = 0.6 * loo_score + 0.15 * ls_score + 0.25 * unc_score
        notes.append("Weights adjusted: LOO upweighted (ls diagnostic unreliable)")
    else:
        trust_score = 0.5 * loo_score + 0.3 * ls_score + 0.2 * unc_score
    trust_score = float(np.clip(trust_score, 0, 1))

    if trust_score >= 0.65:
        trust_flag = "trust"
    elif trust_score >= 0.40:
        trust_flag = "caution"
    else:
        trust_flag = "distrust"

    if verbose:
        _print_summary(trust_score, trust_flag, loo_mean_abs_z, loo_std_z,
                       loo_calibration, ls_fitted, mean_nn_distance, ls_nn_ratio,
                       ls_flag, prior_sigma, posterior_sigma,
                       uncertainty_reduction, unc_flag, n, d, notes)

    return TrustResult(
        trust_score=trust_score, trust_flag=trust_flag,
        loo_mean_abs_z=loo_mean_abs_z, loo_std_z=loo_std_z,
        loo_calibration=loo_calibration, loo_residuals=loo_residuals,
        ls_fitted=ls_fitted, mean_nn_distance=mean_nn_distance,
        ls_nn_ratio=ls_nn_ratio, ls_flag=ls_flag,
        prior_sigma=prior_sigma, posterior_sigma=posterior_sigma,
        uncertainty_reduction=uncertainty_reduction, unc_flag=unc_flag,
        n_obs=n, n_dims=d, notes=notes,
    )


def interpret_trust(result: TrustResult, verbose: bool = True) -> str:
    """Return a human-readable routing recommendation from a TrustResult."""
    lines = [
        f"GP Trust Score: {result.trust_score:.2f} [{result.trust_flag.upper()}]",
        f"  n={result.n_obs}, d={result.n_dims}",
        f"  LOO calibration : {result.loo_calibration:<15} "
        f"mean|z|={result.loo_mean_abs_z:.2f} (ideal ~0.8)",
        f"  ls/nn ratio     : {result.ls_flag:<15} "
        f"ratio={result.ls_nn_ratio:.2f} (ideal >1.0)",
        f"  Uncertainty red.: {result.unc_flag:<15} "
        f"reduction={result.uncertainty_reduction:.2f} (ideal >0.3)",
    ]
    if result.notes:
        lines.append("  Notes:")
        for note in result.notes:
            lines.append(f"    - {note}")

    if result.trust_flag == "trust":
        lines.append("  → Recommendation: USE EGBO or UCB")
    elif result.trust_flag == "caution":
        lines.append("  → Recommendation: USE UCB with high beta, monitor improvement")
    else:
        lines.append("  → Recommendation: USE LHS or random (GP not reliable)")

    return "\n".join(lines)


def _print_summary(trust_score, trust_flag, loo_mean_abs_z, loo_std_z,
                   loo_calibration, ls_fitted, mean_nn_distance, ls_nn_ratio,
                   ls_flag, prior_sigma, posterior_sigma,
                   uncertainty_reduction, unc_flag, n, d, notes):
    print(f"\nGP Trust Diagnostics (n={n}, d={d})")
    print(f"  Trust score: {trust_score:.2f} [{trust_flag.upper()}]")
    print(f"  D1 LOO:  mean|z|={loo_mean_abs_z:.2f}  std(z)={loo_std_z:.2f}"
          f"  → {loo_calibration}")
    print(f"  D2 ls/nn: ls={ls_fitted:.3f}  nn={mean_nn_distance:.3f}"
          f"  ratio={ls_nn_ratio:.2f}  → {ls_flag}")
    print(f"  D3 unc:  prior_σ={prior_sigma:.3f}  post_σ={posterior_sigma:.3f}"
          f"  reduction={uncertainty_reduction:.2f}  → {unc_flag}")
    for note in notes:
        print(f"  ! {note}")


# ── Validation on existing ADA datasets ───────────────────────────────────────

def validate_on_ada_datasets(data_dir: str = "data", n_init: int = 10,
                              rng_seed: int = 42):
    """
    Validate diagnostics on existing ADA datasets.
    For each dataset, run the diagnostic battery on n_init random observations
    and check whether the trust score correctly predicts landscape regime:
      - structured landscapes (coatings, pareto_20210112) → should be TRUST
      - flat landscape (pareto_20201218) → should be CAUTION or DISTRUST

    Also computes post-hoc EGBO advantage to compare against trust score.
    """
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from oracle import NNOracle
    from shared_seed_experiment import generate_shared_inits

    DATASET_MAP = {
        "coatings":        ("coatings",                           "structured"),
        "pareto_20201218": ("pareto_campaign 2020-12-18_17-38-40","flat"),
        "pareto_20210112": ("pareto_campaign 2021-01-12_16-26-56","structured"),
        "hartmann3":       ("hartmann3",                          "structured"),
        "hartmann6":       ("hartmann6",                          "structured"),
    }

    print("="*70)
    print("GP TRUST DIAGNOSTIC VALIDATION ON ADA DATASETS")
    print(f"n_init={n_init}, rng_seed={rng_seed}")
    print("="*70)
    print(f"\n{'Dataset':<20} {'Regime':<12} {'Trust':>6} {'Flag':<10} "
          f"{'LOO|z|':>7} {'ls/nn':>6} {'ΔUncert':>8}")
    print("-"*70)

    results = {}
    for ds_label, (ds_name, regime) in DATASET_MAP.items():
        try:
            oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        except Exception:
            continue

        bounds = oracle.bounds()
        inits  = generate_shared_inits(oracle, 10, n_init, rng_seed=rng_seed)

        scores, flags, loo_vals, ls_vals, unc_vals = [], [], [], [], []
        for X_init, y_init in inits:
            r = gp_trust_check(X_init, y_init, bounds)
            scores.append(r.trust_score)
            flags.append(r.trust_flag)
            loo_vals.append(r.loo_mean_abs_z)
            ls_vals.append(r.ls_nn_ratio)
            unc_vals.append(r.uncertainty_reduction)

        mean_score = np.mean(scores)
        dominant_flag = max(set(flags), key=flags.count)
        correct = (regime == "structured" and dominant_flag == "trust") or \
                  (regime == "flat"       and dominant_flag in ("caution","distrust"))
        marker = "✓" if correct else "✗"

        print(f"  {ds_label:<18} {regime:<12} {mean_score:>6.2f} "
              f"{dominant_flag:<10} {np.mean(loo_vals):>7.2f} "
              f"{np.mean(ls_vals):>6.2f} {np.mean(unc_vals):>8.2f}  {marker}")
        results[ds_label] = {
            "regime": regime, "mean_score": mean_score,
            "dominant_flag": dominant_flag, "correct": correct,
        }

    n_correct = sum(r["correct"] for r in results.values())
    n_total = len(results)
    print(f"\n  Correct predictions: {n_correct}/{n_total}")
    print(f"  Accuracy: {100*n_correct/n_total:.0f}%")
    print()
    print("  Ideal outcome:")
    print("    structured datasets → TRUST  (GP reliable, use EGBO)")
    print("    flat datasets       → CAUTION/DISTRUST  (GP unreliable, use LHS)")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on ADA datasets")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--n_init",   type=int, default=10)
    args = parser.parse_args()

    if args.validate:
        validate_on_ada_datasets(args.data_dir, args.n_init)
    else:
        # Demo on synthetic data
        print("Running demo on synthetic structured landscape...")
        rng = np.random.default_rng(42)
        d = 4
        bounds = np.column_stack([np.zeros(d), np.ones(d)])
        X = rng.random((10, d))
        y = np.sin(3*X[:,0]) + np.cos(2*X[:,1])  # smooth structured

        result = gp_trust_check(X, y, bounds, verbose=True)
        print("\n" + interpret_trust(result))

        print("\n\nRunning demo on flat/noisy landscape...")
        y_flat = rng.standard_normal(10)
        result_flat = gp_trust_check(X, y_flat, bounds, verbose=True)
        print("\n" + interpret_trust(result_flat))
