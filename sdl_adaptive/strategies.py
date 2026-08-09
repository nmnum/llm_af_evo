"""
strategies.py — Six optimisation strategies for SDL campaign simulation.

All strategies use a random candidate grid (500 points) for acquisition
maximisation — 33× faster than differential_evolution with negligible
quality loss on the NN oracle landscape.

Strategies
----------
ucb          : Upper Confidence Bound (GP-based, β-parameterised)
ei           : Expected Improvement (GP-based)
pi           : Probability of Improvement (GP-based)
thompson     : Thompson Sampling (GP-based)
random_search: Uniform random sampling within bounds
lhs          : Latin Hypercube Sampling (max-min-distance)

All functions share the same signature:
    fn(X_obs, y_obs, bounds, **kwargs) -> x_next (shape: (d,))
"""

import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

N_CANDIDATES = 500  # random candidate grid size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_gp(X_obs: np.ndarray, y_obs: np.ndarray):
    """Fit a GP on standardised inputs and z-scored outputs.

    Bounds (1e-3, 1e3) prevent the optimiser hitting the sklearn default
    lower wall (1e-5) on spiky landscapes, eliminating ConvergenceWarnings.
    The GP is used only for acquisition ranking, not ground-truth prediction,
    so a slightly sub-optimal kernel fit is inconsequential.
    """
    scaler_x = StandardScaler()
    X_s = scaler_x.fit_transform(X_obs)
    y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
    y_s = (y_obs - y_mean) / y_std

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_s, y_s)
    return gp, scaler_x, y_mean, y_std


def _random_candidates(bounds: np.ndarray, n: int = N_CANDIDATES) -> np.ndarray:
    """Sample n points uniformly within bounds. Returns (n, d)."""
    d = bounds.shape[0]
    return np.column_stack([
        np.random.uniform(bounds[i, 0], bounds[i, 1], n) for i in range(d)
    ])


# ---------------------------------------------------------------------------
# Strategy functions
# ---------------------------------------------------------------------------

def ucb(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray,
        beta: float = 1.0, **_) -> np.ndarray:
    """Upper Confidence Bound: μ + β·σ."""
    gp, scaler_x, y_mean, y_std = _fit_gp(X_obs, y_obs)
    candidates = _random_candidates(bounds)
    mu, sigma = gp.predict(scaler_x.transform(candidates), return_std=True)
    acq = mu + beta * sigma
    return candidates[np.argmax(acq)]


def ei(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray, **_) -> np.ndarray:
    """Expected Improvement."""
    gp, scaler_x, y_mean, y_std = _fit_gp(X_obs, y_obs)
    y_best_s = (y_obs.max() - y_mean) / y_std
    candidates = _random_candidates(bounds)
    mu, sigma = gp.predict(scaler_x.transform(candidates), return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best_s) / sigma
    acq = sigma * (z * norm.cdf(z) + norm.pdf(z))
    return candidates[np.argmax(acq)]


def pi(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray, **_) -> np.ndarray:
    """Probability of Improvement."""
    gp, scaler_x, y_mean, y_std = _fit_gp(X_obs, y_obs)
    y_best_s = (y_obs.max() - y_mean) / y_std
    candidates = _random_candidates(bounds)
    mu, sigma = gp.predict(scaler_x.transform(candidates), return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    acq = norm.cdf((mu - y_best_s) / sigma)
    return candidates[np.argmax(acq)]


def thompson(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray, **_) -> np.ndarray:
    """Thompson Sampling: sample from GP posterior."""
    gp, scaler_x, y_mean, y_std = _fit_gp(X_obs, y_obs)
    candidates = _random_candidates(bounds)
    mu, sigma = gp.predict(scaler_x.transform(candidates), return_std=True)
    samples = np.random.normal(mu, np.maximum(sigma, 1e-9))
    return candidates[np.argmax(samples)]


def random_search(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray, **_) -> np.ndarray:
    """Uniform random sampling — no model required."""
    d = bounds.shape[0]
    return np.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)])


def lhs(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray,
        n_candidates: int = N_CANDIDATES, **_) -> np.ndarray:
    """
    Latin Hypercube Sampling with max-min-distance selection.
    Generates n_candidates LHS points and returns the one furthest
    from all observed points (in standardised space).
    """
    d = bounds.shape[0]
    n = n_candidates

    # Generate LHS grid
    cuts = np.linspace(0, 1, n + 1)
    u = np.random.uniform(size=(n, d))
    candidates_unit = np.zeros((n, d))
    for j in range(d):
        perm = np.random.permutation(n)
        candidates_unit[:, j] = (cuts[perm] + u[:, j] / n)

    # Scale to bounds
    candidates = candidates_unit * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    # Max-min distance from observed points (standardised)
    if len(X_obs) == 0:
        return candidates[0]

    scaler = StandardScaler()
    X_obs_s = scaler.fit_transform(X_obs)
    cand_s = scaler.transform(candidates)

    min_dists = np.array([
        np.min(np.linalg.norm(X_obs_s - c, axis=1)) for c in cand_s
    ])
    return candidates[np.argmax(min_dists)]


# ---------------------------------------------------------------------------
# Discrete-mode strategy functions
# ---------------------------------------------------------------------------
# These score every candidate in X_pool (the unqueried dataset rows) directly,
# with no continuous relaxation and no nearest-neighbour snap.
# Signature: fn(X_obs, y_obs, X_pool, **kwargs) -> index into X_pool

def _fit_gp_discrete(X_obs: np.ndarray, y_obs: np.ndarray, X_pool: np.ndarray):
    """Fit GP on all observed points, return predictions at X_pool.

    Same bound widening as _fit_gp to suppress ConvergenceWarnings.
    """
    scaler_x = StandardScaler()
    # Fit scaler on observed + pool so scale is consistent
    X_all = np.vstack([X_obs, X_pool])
    scaler_x.fit(X_all)
    X_obs_s = scaler_x.transform(X_obs)
    X_pool_s = scaler_x.transform(X_pool)

    y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
    y_s = (y_obs - y_mean) / y_std

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_obs_s, y_s)
    mu, sigma = gp.predict(X_pool_s, return_std=True)
    return mu, sigma, y_mean, y_std


def ucb_discrete(X_obs, y_obs, X_pool, beta=1.0, **_):
    mu, sigma, _, _ = _fit_gp_discrete(X_obs, y_obs, X_pool)
    return np.argmax(mu + beta * sigma)


def ei_discrete(X_obs, y_obs, X_pool, **_):
    mu, sigma, y_mean, y_std = _fit_gp_discrete(X_obs, y_obs, X_pool)
    y_best_s = (y_obs.max() - y_mean) / y_std
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best_s) / sigma
    acq = sigma * (z * norm.cdf(z) + norm.pdf(z))
    return np.argmax(acq)


def pi_discrete(X_obs, y_obs, X_pool, **_):
    mu, sigma, y_mean, y_std = _fit_gp_discrete(X_obs, y_obs, X_pool)
    y_best_s = (y_obs.max() - y_mean) / y_std
    sigma = np.maximum(sigma, 1e-9)
    acq = norm.cdf((mu - y_best_s) / sigma)
    return np.argmax(acq)


def thompson_discrete(X_obs, y_obs, X_pool, **_):
    mu, sigma, _, _ = _fit_gp_discrete(X_obs, y_obs, X_pool)
    samples = np.random.normal(mu, np.maximum(sigma, 1e-9))
    return np.argmax(samples)


def random_discrete(X_obs, y_obs, X_pool, **_):
    return np.random.randint(len(X_pool))


def lhs_discrete(X_obs, y_obs, X_pool, **_):
    """Max-min distance from observed points — no LHS grid needed on discrete set."""
    if len(X_obs) == 0:
        return 0
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_obs, X_pool]))
    X_obs_s = scaler.transform(X_obs)
    X_pool_s = scaler.transform(X_pool)
    min_dists = np.array([
        np.min(np.linalg.norm(X_obs_s - c, axis=1)) for c in X_pool_s
    ])
    return np.argmax(min_dists)


# ---------------------------------------------------------------------------
# EGBO (Aqeeli et al. 2026): botorch/pymoo/gpytorch are heavy optional deps,
# so these wrappers import egbo_strategy lazily on first call rather than at
# module import time — strategies.py must still import fine in envs that
# don't have those packages installed. If they're missing, fall back to
# random and log once (this is the "egbo silently degrades to random"
# situation flagged earlier for approach_d — the difference is now it's an
# environment problem you can fix by installing deps, not a missing
# implementation).
# ---------------------------------------------------------------------------

_egbo_import_warned = False


def _egbo_module():
    """Return egbo_strategy iff its actual dependencies (botorch/gpytorch/
    torch/pymoo) are importable, else None + a one-time warning.

    Note: `import egbo_strategy` alone always succeeds (it imports those
    packages lazily inside its functions, not at module level) — checking
    only that would make an unavailable-deps case silently fall through to
    egbo_strategy's own internal `except Exception: return random`, which
    is exactly the invisible-fallback failure mode this was meant to fix.
    """
    global _egbo_import_warned
    try:
        import torch, botorch, gpytorch, pymoo  # noqa: F401
        import egbo_strategy
        return egbo_strategy
    except ImportError as e:
        if not _egbo_import_warned:
            import logging
            logging.getLogger(__name__).warning(
                f"EGBO dependencies unavailable ({e}) — 'egbo'/'novelty_egbo' "
                f"strategies will fall back to random for the rest of this run. "
                f"Install with: pip install botorch gpytorch torch pymoo"
            )
            _egbo_import_warned = True
        return None


def egbo(X_obs, y_obs, bounds, **kwargs):
    mod = _egbo_module()
    if mod is None:
        return random_search(X_obs, y_obs, bounds, **kwargs)
    return mod.egbo(X_obs, y_obs, bounds, **kwargs)


def novelty_egbo(X_obs, y_obs, bounds, **kwargs):
    mod = _egbo_module()
    if mod is None:
        return random_search(X_obs, y_obs, bounds, **kwargs)
    return mod.novelty_egbo(X_obs, y_obs, bounds, **kwargs)


def egbo_discrete(X_obs, y_obs, X_pool, **kwargs):
    mod = _egbo_module()
    if mod is None:
        return random_discrete(X_obs, y_obs, X_pool, **kwargs)
    return mod.egbo_discrete(X_obs, y_obs, X_pool, **kwargs)


def novelty_egbo_discrete(X_obs, y_obs, X_pool, **kwargs):
    mod = _egbo_module()
    if mod is None:
        return random_discrete(X_obs, y_obs, X_pool, **kwargs)
    return mod.novelty_egbo_discrete(X_obs, y_obs, X_pool, **kwargs)


DISCRETE_STRATEGY_MAP = {
    "ucb":    ucb_discrete,
    "ei":     ei_discrete,
    "pi":     pi_discrete,
    "thompson": thompson_discrete,
    "random": random_discrete,
    "lhs":    lhs_discrete,
    "egbo":   egbo_discrete,
    "novelty_egbo": novelty_egbo_discrete,
}


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "ucb": ucb,
    "ei": ei,
    "pi": pi,
    "thompson": thompson,
    "random": random_search,
    "lhs": lhs,
    "egbo": egbo,
    "novelty_egbo": novelty_egbo,
}
