"""
egbo_strategy.py — single-step EGBO adapter for STRATEGY_MAP / DISCRETE_STRATEGY_MAP.

shared_seed_experiment.py's run_egbo_campaign() is a self-contained *batch*
campaign runner (own X_obs/y_obs state, batch_size=4 points per round,
n_batches loop). It is not the same signature as the strategies in
strategies.py — `fn(X_obs, y_obs, bounds, **kwargs) -> x_next` (continuous) or
`fn(X_obs, y_obs, X_pool, **kwargs) -> pool_idx` (discrete) — which is what
approach_d's controller (and any STRATEGY_MAP-driven loop) actually calls.

This module re-implements the same recipe (Aqeeli et al. 2026: BoTorch
SingleTaskGP + qLogNoisyExpectedImprovement, pymoo U-NSGA-III evolutionary
candidates seeded from the best observed points, novelty-aware greedy
selection) as a *stateless single-point* call, so it fits the same interface
every other strategy in strategies.py already uses. Fit/acquisition-optimise
cost is paid every call rather than once per 4-step batch — that's the
necessary trade-off of a per-step controller architecture, same as
ucb/ei/pi/thompson already refitting a GP from scratch every call.

Requires: torch, botorch, gpytorch, pymoo (see shared_seed_experiment.py's
module docstring for the install line). Import is lazy so nothing else in
this package needs these to import strategies.py.
"""

import logging
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _egbo_select_point(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    bounds: np.ndarray,
    merit_weight: float = 1.0,
    evo_candidates: int = 72,
    qnehvi_candidates: int = 8,
    random_state: int = 0,
) -> np.ndarray:
    """Core EGBO point selection, in the ORIGINAL (unnormalised) input space.

    Returns a single (d,) point. Callers decide what to do with it (return
    raw for continuous strategies, snap to nearest unqueried pool row for
    discrete ones) — mirrors how run_egbo_campaign snaps AFTER selection.
    """
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.transforms import normalize, unnormalize
    from botorch.optim.optimize import optimize_acqf
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    warnings.filterwarnings("ignore")
    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    d = bounds.shape[0]
    bounds_t = torch.tensor(bounds.T, **tkwargs)  # (2, d)
    standard_bounds = torch.zeros(2, d, **tkwargs)
    standard_bounds[1] = 1.0

    train_x = torch.tensor(np.asarray(X_obs, dtype=float), **tkwargs)
    train_y = torch.tensor(np.asarray(y_obs, dtype=float).reshape(-1, 1), **tkwargs)
    train_x_norm = normalize(train_x, bounds_t)

    gp_model = SingleTaskGP(train_x_norm, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll, max_retries=1, options={"maxiter": 50})

    import inspect as _inspect
    _nei_params = list(_inspect.signature(qLogNoisyExpectedImprovement.__init__).parameters)
    _acq_kwargs = dict(
        model=gp_model,
        X_baseline=train_x_norm,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    )
    if "prune_baseline" in _nei_params:
        _acq_kwargs["prune_baseline"] = True
    if "cache_root" in _nei_params:
        _acq_kwargs["cache_root"] = True
    acq_fn = qLogNoisyExpectedImprovement(**_acq_kwargs)

    try:
        qbo_x_norm, _ = optimize_acqf(
            acq_fn, bounds=standard_bounds, q=qnehvi_candidates,
            num_restarts=1, raw_samples=16,
            options={"batch_limit": qnehvi_candidates, "maxiter": 20},
        )
    except Exception:
        from botorch.utils.sampling import draw_sobol_samples
        qbo_x_norm = draw_sobol_samples(bounds=standard_bounds, n=qnehvi_candidates, q=1).squeeze(-2)

    top_k = min(evo_candidates, train_x_norm.shape[0])
    top_idx = train_y.squeeze(-1).argsort(descending=True)[:top_k]
    seed_x = train_x_norm[top_idx].cpu().numpy()
    if seed_x.shape[0] < evo_candidates:
        rng_pad = np.random.default_rng(random_state)
        pad = rng_pad.random((evo_candidates - seed_x.shape[0], d))
        seed_x = np.vstack([seed_x, pad])

    try:
        ref_dirs = get_reference_directions("energy", 1, evo_candidates, seed=random_state)
    except Exception:
        rng = np.random.default_rng(random_state)
        ref_dirs = rng.random((evo_candidates, 1))

    try:
        algo = UNSGA3(pop_size=evo_candidates, ref_dirs=ref_dirs, sampling=seed_x)
        pm = PymooProblem(n_var=d, n_obj=1, n_constr=0, xl=np.zeros(d), xu=np.ones(d))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop_size_actual = len(pop)
        f_seed = train_y.squeeze(-1).argsort(descending=True)
        f_vals = -train_y[f_seed].cpu().numpy()
        if len(f_vals) >= pop_size_actual:
            pop.set("F", f_vals[:pop_size_actual].reshape(-1, 1))
        else:
            pad = np.full((pop_size_actual - len(f_vals), 1), f_vals[-1])
            pop.set("F", np.vstack([f_vals.reshape(-1, 1), pad]))
        algo.tell(infills=pop)
        ea_x_norm = torch.tensor(algo.ask().get("X"), **tkwargs)
    except Exception:
        ea_x_norm = torch.rand(evo_candidates, d, **tkwargs)

    candidates_norm = torch.cat([qbo_x_norm, ea_x_norm], dim=0)
    with torch.no_grad():
        try:
            acq_vals = [float(acq_fn(candidates_norm[i].unsqueeze(0)).item())
                        for i in range(candidates_norm.shape[0])]
        except Exception:
            acq_vals = list(np.random.randn(candidates_norm.shape[0]))

    sel_idx = _select_novelty_single(train_x_norm, candidates_norm, acq_vals, merit_weight=merit_weight)
    sel_x_norm = candidates_norm[sel_idx].cpu().numpy()
    x_raw = unnormalize(torch.tensor(sel_x_norm, **tkwargs).unsqueeze(0), bounds_t).cpu().numpy()[0]
    return x_raw


def _select_novelty_single(train_x_norm, candidates_norm, acq_vals, merit_weight=0.7):
    """Same scoring as shared_seed_experiment._select_novelty, batch_size=1."""
    import torch
    merit_weight = float(np.clip(merit_weight, 0.0, 1.0))
    acq = np.asarray(acq_vals, dtype=float)
    finite = np.isfinite(acq)
    merit_norm = np.zeros_like(acq)
    if finite.any():
        lo, hi = float(np.min(acq[finite])), float(np.max(acq[finite]))
        merit_norm[finite] = (acq[finite] - lo) / (hi - lo) if hi - lo > 1e-12 else 1.0

    nov = torch.cdist(candidates_norm, train_x_norm).min(dim=1).values.cpu().numpy()
    n_lo, n_hi = float(np.min(nov)), float(np.max(nov))
    nov_norm = np.ones_like(nov) if n_hi - n_lo <= 1e-12 else (nov - n_lo) / (n_hi - n_lo)
    score = merit_weight * merit_norm + (1.0 - merit_weight) * nov_norm
    return int(np.argmax(score))


def egbo(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray,
         merit_weight: float = 1.0, **_) -> np.ndarray:
    """Continuous-space EGBO: fn(X_obs, y_obs, bounds, **kwargs) -> x_next.
    merit_weight=1.0 (standard EGBO, pure acquisition) is the default so
    STRATEGY_MAP["egbo"] matches Aqeeli et al.'s merit_weight=1.0 variant;
    novelty_egbo below uses 0.7."""
    d = bounds.shape[0]
    n = len(X_obs)
    if n <= d + 1:
        # Too few observations for a stable GP fit — same guard the other
        # strategies use implicitly via their random-candidate fallback.
        return np.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)])
    try:
        x_next = _egbo_select_point(X_obs, y_obs, bounds, merit_weight=merit_weight)
        return np.clip(x_next, bounds[:, 0], bounds[:, 1])
    except Exception as e:
        logger.warning(f"egbo point selection failed ({e!r}) — falling back to random for this step")
        return np.array([np.random.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)])


def novelty_egbo(X_obs: np.ndarray, y_obs: np.ndarray, bounds: np.ndarray, **kwargs) -> np.ndarray:
    kwargs.pop("merit_weight", None)
    return egbo(X_obs, y_obs, bounds, merit_weight=0.7, **kwargs)


def egbo_discrete(X_obs, y_obs, X_pool, merit_weight: float = 1.0, **_) -> int:
    """Discrete-pool EGBO: fn(X_obs, y_obs, X_pool, **kwargs) -> pool_idx.
    bounds are derived from X_pool's min/max per dimension (mirrors how
    run_egbo_campaign snaps into the dataset's own coordinate range)."""
    d = X_pool.shape[1]
    n = len(X_obs)
    if n <= d + 1 or len(X_pool) == 0:
        return int(np.random.randint(len(X_pool)))
    bounds = np.column_stack([X_pool.min(axis=0), X_pool.max(axis=0)])
    # Degenerate dims (all pool rows identical on that axis) break normalize().
    flat = bounds[:, 1] <= bounds[:, 0]
    if flat.any():
        bounds[flat, 1] = bounds[flat, 0] + 1e-6
    try:
        x_next = _egbo_select_point(X_obs, y_obs, bounds, merit_weight=merit_weight)
        dists = np.linalg.norm(X_pool - x_next, axis=1)
        return int(np.argmin(dists))
    except Exception as e:
        logger.warning(f"egbo_discrete point selection failed ({e!r}) — falling back to random for this step")
        return int(np.random.randint(len(X_pool)))


def novelty_egbo_discrete(X_obs, y_obs, X_pool, **kwargs) -> int:
    kwargs.pop("merit_weight", None)
    return egbo_discrete(X_obs, y_obs, X_pool, merit_weight=0.7, **kwargs)
