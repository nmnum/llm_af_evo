"""
baseline_decomposition_strategies.py — decomposing `mo_egbo_novelty` ITSELF
the same way the eight-gate arc decomposed L's evolved AFs. Every result in
`L_COATINGS_FINDINGS.md` compares against the whole composite (qLogNEHVI +
UNSGA3 evolutionary generation + novelty-weighted selection) without ever
checking whether each piece is pulling its weight in THIS project's regime
(16D categorical / discrete real-data snap-query, short-horizon 10-40 obs,
noisy) — as opposed to the continuous synthetic/materials benchmarks a
method like qNEHVI+evolution+novelty is more commonly validated against in
the literature. "Ties the best composite" is only as strong a claim as
"best" is established to be, in this exact regime.

Three new strategies, plus two that already existed and are reused as-is
(`strategy_mo_random`, the required floor; `strategy_mo_egbo_real`, which
already isolates exactly the novelty-selection contribution — qLogNEHVI +
UNSGA3 + top-k, no novelty weighting — and is now direction-fixed to work
correctly on coatings, see excipient_campaign_mo.py's recent fix):

  strategy_qnehvi_plain — qLogNEHVI ALONE. No UNSGA3 evolutionary
    candidates, no novelty-weighted selection, no extra candidate pool at
    all: optimize_acqf(q=batch_size) already returns exactly batch_size
    jointly-optimised points, which are returned directly. This is the
    textbook/minimal form of the "joint MC-integrated batch acquisition"
    mechanism the whole eight-gate arc's design rule is built around — if
    THIS ties mo_egbo_novelty, the evolutionary generation and novelty
    selection layered on top aren't earning their keep in this regime
    either, which sharpens the design rule from "decomposing joint MC
    integration loses" to "adding sophistication anywhere, including
    inside what's been called the baseline, doesn't help here."

  strategy_unsga3_plain — pure evolutionary, NO GP/acquisition component
    at all. UNSGA3 evolves a population seeded from the current observed
    Pareto front (same seeding logic as every qLogNEHVI+UNSGA3 strategy's
    ea_x construction) and the evolved population IS the next batch
    directly — no surrogate model, no acquisition function, no MC
    integration anywhere. Tests whether the Bayesian/surrogate half of
    the pipeline is contributing anything distinguishable from pure
    evolutionary search in this regime — a different question from
    anything the scoring/selection/generation/composition gates asked.

  strategy_parego — Chebyshev-scalarised UCB, a genuinely weak, simple,
    well-known standard MOBO baseline (ParEGO-style): a fresh random
    objective-weight vector each batch, a single scalarised acquisition
    value per candidate (weighted-sum posterior mean + UCB bonus), top-k
    selection from a standard perturb-and-explore candidate pool. Given
    the literature finding that explicit HV/Pareto-aware methods
    "almost universally" beat scalarisation, this should lose clearly —
    if it DOESN'T, that's a more alarming finding: these two domains, at
    this budget, wouldn't discriminate between MOBO methods at all,
    undercutting the interpretability of every result in the eight-gate
    arc, including the original mAb tie.

All three take the same (oracle, X_obs, Y_obs, bounds, batch_size, rng,
**kw) signature as every other strategy_fn in this project, are
directions-aware via oracle.objective_directions() (not the module-level
OBJECTIVE_DIRECTIONS constant — the exact bug just fixed in
strategy_mo_egbo_real), and fall back to strategy_mo_egbo (with the
failure reason surfaced, not swallowed) on any exception, matching every
other strategy's established resilience pattern.
"""

import pathlib
import sys
import warnings

import numpy as np
import torch

_LLM_AF_EVO = pathlib.Path(__file__).resolve().parent
while _LLM_AF_EVO.name != "llm_af_evo":
    _LLM_AF_EVO = _LLM_AF_EVO.parent
_ROOT = _LLM_AF_EVO.parent
for _p in (
    _ROOT,
    _LLM_AF_EVO / "shared",
    _LLM_AF_EVO / "v1_pre_v2" / "src",
    _LLM_AF_EVO / "v1_pre_v2" / "experiments",
    _LLM_AF_EVO / "v2" / "src",
    _LLM_AF_EVO / "v2" / "experiments",
):
    sys.path.insert(0, str(_p))

from excipient_campaign_mo import pareto_front_of, strategy_mo_egbo, _make_pool


def strategy_qnehvi_plain(oracle, X_obs, Y_obs, bounds, batch_size, rng, **kw):
    """qLogNEHVI alone — no UNSGA3, no novelty selection, no extra pool.
    optimize_acqf(q=batch_size) IS the batch."""
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import unnormalize
        from botorch.optim.optimize import optimize_acqf

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        directions = oracle.objective_directions()

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1.0

        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        models = [SingleTaskGP(train_x, train_y[:, j:j + 1], outcome_transform=Standardize(m=1))
                  for j in range(M)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        qbo_x, _ = optimize_acqf(
            acq_fn, bounds=standard_bounds, q=batch_size,
            num_restarts=2, raw_samples=16, options={"maxiter": 20},
        )
        new_x_raw = (unnormalize(qbo_x, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())
        return new_x_raw, {"qnehvi_plain": True}

    except Exception as e:
        warnings.warn(f"strategy_qnehvi_plain failed ({e}), falling back to "
                       f"lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["qnehvi_plain"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


def strategy_unsga3_plain(oracle, X_obs, Y_obs, bounds, batch_size, rng, **kw):
    """Pure evolutionary — no GP, no acquisition function, no MC integration
    anywhere. UNSGA3's evolved offspring, seeded from the current Pareto
    front, ARE the next batch directly."""
    try:
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        directions = oracle.objective_directions()

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        pop_candidates = max(batch_size, 20)

        pf_idx_np = pareto_front_of(Y_obs, directions=directions)
        seed_pool_idx = pf_idx_np if len(pf_idx_np) > 0 else np.arange(len(Y_obs))
        seed_x = X_norm[seed_pool_idx]
        if seed_x.shape[0] < pop_candidates:
            pad = rng.random((pop_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        seed_x = seed_x[:max(pop_candidates, 2)]

        ref_dirs = get_reference_directions("energy", M, pop_candidates,
                                             seed=int(rng.integers(1e6)))
        pop_size = max(len(ref_dirs), pop_candidates, 2)
        algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x,
                       seed=int(rng.integers(1e6)))
        pm = PymooProblem(n_var=d, n_obj=M, n_constr=0, xl=np.zeros(d), xu=np.ones(d))
        algo.setup(pm, termination=NoTermination())
        pop = algo.ask()
        pop_size_actual = len(pop)
        f_idx = seed_pool_idx[:pop_size_actual] if \
            len(seed_pool_idx) >= pop_size_actual else seed_pool_idx
        f_vals = -Y_int[f_idx]  # pymoo minimises
        if len(f_vals) >= pop_size_actual:
            pop.set("F", f_vals[:pop_size_actual])
        else:
            pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
            pop.set("F", np.vstack([f_vals, pad]))
        algo.tell(infills=pop)
        evolved_x_n = algo.ask().get("X")

        # No acquisition/surrogate to rank the evolved population by — take
        # a diverse subsample (shuffle then truncate) rather than the
        # first batch_size, same "don't silently favour array position"
        # fix strategy_mo_egbo already applies for the same reason.
        idx = np.arange(len(evolved_x_n))
        rng.shuffle(idx)
        selected_n = evolved_x_n[idx[:batch_size]]
        selected_raw = selected_n * (hi - lo) + lo
        return selected_raw, {"unsga3_plain": True}

    except Exception as e:
        warnings.warn(f"strategy_unsga3_plain failed ({e}), falling back to "
                       f"lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["unsga3_plain"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


def strategy_parego(oracle, X_obs, Y_obs, bounds, batch_size, rng, beta: float = 2.0, **kw):
    """Chebyshev-scalarised UCB (ParEGO-style) — a genuinely weak, simple
    standard MOBO baseline. Fresh random objective weights each batch,
    fits one independent GP per objective (same GP-fitting code as every
    other strategy here, so any difference is attributable to the
    scalarisation+single-score-selection idea, not surrogate quality),
    scores a standard perturb-and-explore candidate pool by weighted-sum
    UCB, top-k selects."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        directions = oracle.objective_directions()
        lo, hi = bounds[:, 0], bounds[:, 1]
        M = Y_obs.shape[1]
        X_obs_n = (X_obs - lo) / (hi - lo + 1e-12)

        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]
        # Per-objective z-score, so the Chebyshev weights combine
        # comparable scales instead of being dominated by whichever
        # objective happens to have the largest raw numeric range.
        Y_mean, Y_std = Y_int.mean(axis=0), Y_int.std(axis=0) + 1e-9
        Y_z = (Y_int - Y_mean) / Y_std

        pool_n = _make_pool(X_obs_n, Y_obs, bounds, rng, n=60, directions=directions)

        weights = rng.dirichlet(np.ones(M))  # fresh random simplex weights each call

        mu_all = np.zeros((len(pool_n), M))
        sigma_all = np.zeros((len(pool_n), M))
        for j in range(M):
            kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                           n_restarts_optimizer=1,
                                           random_state=int(rng.integers(1e6)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X_obs_n, Y_z[:, j])
            mu, sigma = gp.predict(pool_n, return_std=True)
            mu_all[:, j] = mu
            sigma_all[:, j] = sigma

        # Chebyshev (weighted-Tchebycheff) scalarisation, not plain
        # weighted sum — the standard ParEGO choice, since weighted sum
        # can't reach concave regions of a Pareto front even in principle.
        ucb = mu_all + beta * sigma_all
        scalar = (weights[None, :] * (Y_z.max(axis=0)[None, :] - ucb)).max(axis=1)
        scores = -scalar  # smaller Chebyshev distance-to-ideal = better

        k = min(batch_size, len(pool_n))
        selected = np.argsort(scores)[-k:]
        selected_raw = pool_n[selected] * (hi - lo) + lo
        return selected_raw, {"parego": True, "weights": weights.tolist()}

    except Exception as e:
        warnings.warn(f"strategy_parego failed ({e}), falling back to "
                       f"lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["parego"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra
