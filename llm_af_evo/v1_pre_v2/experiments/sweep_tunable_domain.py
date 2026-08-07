"""
sweep_tunable_domain.py — cheap grid sweep over TunableSyntheticMOOracle's
knobs to find a (plateau_sharpness, noise_level) working point that is
NOT mu_sum-dominated and NOT noise-floor-dominated, BEFORE spending LLM-
evolution compute on it. Costs a handful of GP fits, no LLM calls, no full
BO campaigns — this is the "cleaner substrate" selection step, done
empirically instead of by guessing a single value.

What it measures per grid cell
-------------------------------
1. dominance_ratio = std(mu_sum) / std(sigma_norm) across the candidate
   pool, using a GP fit on a small random training set (n=30, matching the
   coatings/mAb campaign-scale regime). HIGH ratio -> mu_sum already
   separates the pool -> sigma term (normalised or not) can't move the
   ranking -> coatings' failure mode. We want this LOW enough that sigma
   has room to matter, i.e. comparable in magnitude to 1, not >>1.
2. achieved_cv = mean(obs_std / |f_true|) across queried points, the
   realised coefficient of variation given noise_level/noise_mode — a
   sanity check that the knob produces what it claims (noise_level is a
   nominal setting, not always the realised CV, especially for
   input_dependent). We want this well under mAb's ~0.497, ideally <0.15.
3. front_range_ratio = front_range[f2] / front_range[f1] — confirms the
   two objectives are actually on different scales in the observed data
   (not just nominally, since scale2 could still yield a similar realised
   spread depending on plateau_sharpness).

Usage: python sweep_tunable_domain.py
Prints a table; pick the (plateau_sharpness, noise_level) cell with
dominance_ratio closest to 1 and achieved_cv comfortably low, then pass
those as the domain's fixed parameters for the real evolution run.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from tunable_synthetic_oracle import TunableSyntheticMOOracle  # noqa: E402

N_TRAIN = 30
N_REPS = 5


def _fit_gp(X_train, y_train):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(X_train, y_train)
    return gp


def _pareto_front_range(Y_obs: np.ndarray) -> np.ndarray:
    """Range (max-min) of each objective within the non-dominated set of
    Y_obs, both-minimise convention — same definition score_pool's
    pareto_front_range context field uses."""
    n = len(Y_obs)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(Y_obs[j] <= Y_obs[i]) and np.any(Y_obs[j] < Y_obs[i]):
                is_dominated[i] = True
                break
    front = Y_obs[~is_dominated]
    if len(front) < 2:
        return np.ptp(Y_obs, axis=0)
    return np.ptp(front, axis=0)


def evaluate_cell(plateau_sharpness: float, noise_level: float, noise_mode: str,
                   seed: int) -> dict:
    oracle = TunableSyntheticMOOracle.build(
        plateau_sharpness=plateau_sharpness, noise_level=noise_level,
        noise_mode=noise_mode, seed=seed)
    rng = np.random.default_rng(seed + 100)
    train_idx = rng.choice(len(oracle), size=N_TRAIN, replace=False)

    X_train, Y_train_noisy, cv_samples = [], [], []
    for idx in train_idx:
        x = oracle._X_raw[idx]
        y_noisy, chosen = oracle.query_mo(x)
        f_true = oracle.true_y(chosen)
        X_train.append(x)
        Y_train_noisy.append(y_noisy)
        cv_samples.append(np.abs(y_noisy - f_true) / (np.abs(f_true) + 1e-9))
    X_train = np.array(X_train)
    Y_train_noisy = np.array(Y_train_noisy)

    gp1 = _fit_gp(X_train, Y_train_noisy[:, 0])
    gp2 = _fit_gp(X_train, Y_train_noisy[:, 1])

    remaining = [i for i in range(len(oracle)) if i not in oracle._queried]
    X_pool = oracle._X_raw[remaining]
    mu1, std1 = gp1.predict(X_pool, return_std=True)
    mu2, std2 = gp2.predict(X_pool, return_std=True)

    front_range = _pareto_front_range(Y_train_noisy)
    front_range = np.clip(front_range, 1e-6, None)

    mu_sum = mu1 + mu2
    sigma_norm = std1 / front_range[0] + std2 / front_range[1]

    dominance_ratio = np.std(mu_sum) / (np.std(sigma_norm) + 1e-9)
    achieved_cv = float(np.mean(cv_samples))
    front_range_ratio = float(front_range[1] / front_range[0])

    return dict(plateau_sharpness=plateau_sharpness, noise_level=noise_level,
                noise_mode=noise_mode, dominance_ratio=dominance_ratio,
                achieved_cv=achieved_cv, front_range_ratio=front_range_ratio)


def main():
    # scale2 defaults to 3.0 in TunableSyntheticMOOracle.build now — an
    # earlier sweep at scale2=20 found dominance_ratio in the hundreds even
    # at plateau_sharpness=3, because raw mu_sum isn't scale-invariant the
    # way sigma_norm is. Grid below targets the regime that sweep found
    # workable (dominance_ratio ~15-35) at the new default scale2.
    plateau_grid = [1.0, 2.0, 3.0, 5.0]
    noise_grid = [0.03, 0.08, 0.15]
    noise_mode = "proportional"

    print(f"{'plateau':>8} {'noise_lvl':>10} {'dom_ratio':>10} {'ach_cv':>8} {'range_ratio':>12}")
    rows = []
    for plateau in plateau_grid:
        for noise_level in noise_grid:
            reps = [evaluate_cell(plateau, noise_level, noise_mode, seed=42 + r)
                     for r in range(N_REPS)]
            dom = np.mean([r["dominance_ratio"] for r in reps])
            cv = np.mean([r["achieved_cv"] for r in reps])
            rr = np.mean([r["front_range_ratio"] for r in reps])
            rows.append((plateau, noise_level, dom, cv, rr))
            print(f"{plateau:8.1f} {noise_level:10.2f} {dom:10.2f} {cv:8.3f} {rr:12.2f}")

    print("\nTarget: dominance_ratio near 1 (not >>1 — that's mu_sum dominance,")
    print("coatings' failure mode), achieved_cv well under 0.497 (mAb's failure")
    print("mode; aim <0.15), front_range_ratio far from 1 (confirms objectives")
    print("are on genuinely different scales, the condition front-range")
    print("normalisation needs to have any effect to demonstrate).")


if __name__ == "__main__":
    main()
