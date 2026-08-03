"""
evolutionary_candidates.py — Pre-built diversity candidate generator for LLM-C sandbox.

Provides evolutionary_candidates() using U-NSGA-III (pymoo), matching the
candidate generation step in EGBO (Aqeeli et al. 2026).

This module is injected into the approach_c sandbox so the LLM can call it
without reimplementing evolutionary search from scratch.

Usage in LLM-generated suggest() functions:
    from evolutionary_candidates import evolutionary_candidates
    
    candidates = evolutionary_candidates(X_obs, y_obs, bounds, n=72)
    # candidates: (n, d) array of diverse candidate points within bounds
    # Combine with acquisition-function candidates before selecting x_next
"""

import numpy as np
import warnings


def evolutionary_candidates(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    bounds: np.ndarray,
    n: int = 72,
    random_state: int = 0,
) -> np.ndarray:
    """
    Generate n diverse candidate points using U-NSGA-III evolutionary search.
    
    Seeds the evolutionary population from the best observed points, then
    runs one generation to produce a diverse candidate pool covering the
    input space. Returned candidates are within bounds.

    Parameters
    ----------
    X_obs   : (n_obs, d) observed input points
    y_obs   : (n_obs,) observed outputs (higher is better)
    bounds  : (d, 2) array of [min, max] per dimension
    n       : number of candidates to generate (default 72)
    random_state : random seed

    Returns
    -------
    candidates : (n, d) array of candidate points within bounds
    
    Notes
    -----
    Falls back to LHS if pymoo is unavailable or fails.
    """
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]

    # Normalise to [0,1] for pymoo
    def norm(X):
        return (X - lo) / (hi - lo + 1e-12)

    def denorm(X):
        return X * (hi - lo) + lo

    try:
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        # Seed from top-k observed points
        top_k = min(n, len(X_obs))
        top_idx = np.argsort(y_obs)[-top_k:][::-1]
        seed_x = norm(X_obs[top_idx])

        # Pad to full population size with LHS if needed
        if seed_x.shape[0] < n:
            rng = np.random.default_rng(random_state)
            n_pad = n - seed_x.shape[0]
            # LHS padding
            cuts = np.linspace(0, 1, n_pad + 1)
            u = rng.random((n_pad, d))
            pad = np.zeros((n_pad, d))
            for j in range(d):
                perm = rng.permutation(n_pad)
                pad[:, j] = cuts[perm] + u[:, j] / n_pad
            seed_x = np.vstack([seed_x, np.clip(pad, 0, 1)])

        try:
            ref_dirs = get_reference_directions("energy", 1, n, seed=random_state)
        except Exception:
            ref_dirs = np.random.default_rng(random_state).random((n, 1))

        algo = UNSGA3(pop_size=n, ref_dirs=ref_dirs, sampling=seed_x)
        pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                          xl=np.zeros(d), xu=np.ones(d))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()

        # Set fitness as negative y (minimisation in pymoo)
        y_top = y_obs[top_idx]
        f_vals = -y_top[:n] if len(y_top) >= n else np.pad(-y_top, (0, n - len(y_top)))
        pop_size_actual = len(pop)
        if len(f_vals) >= pop_size_actual:
            pop.set("F", f_vals[:pop_size_actual].reshape(-1, 1))
        else:
            pad = np.full(pop_size_actual - len(f_vals), f_vals[-1] if len(f_vals) else 0.0)
            pop.set("F", np.append(f_vals, pad).reshape(-1, 1))
        algo.tell(infills=pop)

        next_pop = algo.ask()
        candidates_norm = np.clip(next_pop.get("X"), 0, 1)
        return denorm(candidates_norm)

    except Exception:
        # Fallback: Latin Hypercube Sampling
        return _lhs_candidates(bounds, n, random_state)


def _lhs_candidates(bounds: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """LHS fallback — max-min-distance variant."""
    d = bounds.shape[0]
    rng = np.random.default_rng(seed)
    cuts = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    candidates_unit = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        candidates_unit[:, j] = cuts[perm] + u[:, j] / n
    lo, hi = bounds[:, 0], bounds[:, 1]
    return candidates_unit * (hi - lo) + lo


def novelty_select(
    candidates: np.ndarray,
    X_obs: np.ndarray,
    bounds: np.ndarray,
    acq_scores: np.ndarray,
    merit_weight: float = 0.7,
) -> int:
    """
    Select the single best candidate using novelty-aware scoring.
    
    Score_i = w * merit_norm_i + (1-w) * novelty_norm_i
    where novelty = min Euclidean distance in normalised space to any observed point.
    
    Parameters
    ----------
    candidates  : (n_cand, d) candidate points
    X_obs       : (n_obs, d) already-observed points
    bounds      : (d, 2) for normalisation
    acq_scores  : (n_cand,) acquisition function scores (higher is better)
    merit_weight: weight on acquisition vs novelty (default 0.7, paper value)
    
    Returns
    -------
    int : index into candidates of the best point
    """
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo + 1e-12

    cand_norm = (candidates - lo) / span
    obs_norm  = (X_obs - lo) / span

    # Normalise acquisition scores
    a = np.asarray(acq_scores, dtype=float)
    finite = np.isfinite(a)
    merit = np.zeros_like(a)
    if finite.any():
        lo_a, hi_a = a[finite].min(), a[finite].max()
        merit[finite] = (a[finite] - lo_a) / (hi_a - lo_a) if hi_a > lo_a else 1.0

    # Novelty: min distance to observed set
    nov = np.array([
        np.min(np.linalg.norm(obs_norm - c, axis=1))
        for c in cand_norm
    ])
    lo_n, hi_n = nov.min(), nov.max()
    nov_norm = (nov - lo_n) / (hi_n - lo_n) if hi_n > lo_n else np.ones_like(nov)

    score = merit_weight * merit + (1.0 - merit_weight) * nov_norm
    return int(np.argmax(score))
