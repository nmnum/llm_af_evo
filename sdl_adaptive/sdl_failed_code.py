"""
optimiser_template_evo.py — Starting template for Approach C with evolutionary candidates.

This template already uses evolutionary_candidates() as the candidate generation step.
The LLM's job is to improve the acquisition scoring and selection logic,
not to decide whether to use evolutionary diversity.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from evolutionary_candidates import evolutionary_candidates, novelty_select


def suggest(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Suggest the next experiment point.
    Uses evolutionary candidates for diversity + GP acquisition for exploitation.
    """
    d = bounds.shape[0]
    n = len(X_obs)

    # ALWAYS initialise a fallback random point
    x_next = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

    try:
        # Early campaign: space-filling only
        if n <= d + 1:
            return x_next

        # Generate diverse candidates via evolutionary search (seeded from best observed)
        evo_cands = evolutionary_candidates(X_obs, y_obs, bounds, n=72)

        # Generate GP-based candidates for exploitation
        model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=2,
        )
        model.fit(X_obs, y_obs)

        # Score evolutionary candidates with UCB acquisition
        mu, sigma = model.predict(evo_cands, return_std=True)
        beta = 2.0  # exploration weight — adjust based on campaign progress
        acq_scores = mu + beta * sigma

        # Also score a GP-optimised candidate set
        n_gp = 100
        gp_cands = np.column_stack([
            np.random.uniform(lo, hi, n_gp) for lo, hi in bounds
        ])
        mu_gp, sigma_gp = model.predict(gp_cands, return_std=True)
        acq_gp = mu_gp + beta * sigma_gp

        # Merge pools
        all_cands = np.vstack([evo_cands, gp_cands])
        all_scores = np.concatenate([acq_scores, acq_gp])

        # Select with novelty awareness to avoid re-sampling explored regions
        best_idx = novelty_select(all_cands, X_obs, bounds, all_scores, merit_weight=0.7)
        x_next = all_cands[best_idx]

    except Exception:
        pass

    return x_next.flatten()
