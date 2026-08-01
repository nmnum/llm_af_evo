"""
compose_strategies.py — the 2x2 ablation's four strategy cells, built from
ONE shared candidate-generation/model-fitting code path so a result is
attributable to exactly one lever (surrogate choice: independent per-
objective GPs vs. DA-COREG's coregionalized multi-task GP; selection
mechanism: the baseline's own qLogNEHVI-score + novelty-weighted selection
vs. a jointly-composed compose_batch), never a difference in unrelated
plumbing between cells.

                        independent GPs                  DA-COREG
baseline selection      strategy_mo_egbo_novelty          strategy_ablation_cell(
                        (existing, unmodified,             use_da_coreg=True,
                         reused directly — see              use_compose=False)
                         run_synthetic_compose_pilot.py)
compose_batch           strategy_ablation_cell(            strategy_ablation_cell(
                        use_da_coreg=False,                 use_da_coreg=True,
                        use_compose=True)                   use_compose=True)

Candidate generation (qLogNEHVI-optimised qbo_x + UNSGA3-evolved ea_x) is
IDENTICAL across all four cells regardless of which surrogate or selection
mechanism is used — same reasoning as strategy_evolved_af's docstring:
this is what makes the ablation's cells comparable, and it's also the
expensive part, so results aren't attributable to candidates being drawn
from a differently-distributed pool.
"""

import pathlib
import sys
import warnings

import numpy as np
import torch

# Self-sufficient sys.path bootstrap, matching full_replay.py's pattern —
# don't rely on the importing script having already inserted these paths
# (run_synthetic_compose_pilot.py and debug_compose.py both do, but this
# module shouldn't silently depend on import order elsewhere doing so too).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from excipient_campaign_mo import pareto_front_of
from novelty_selection import novelty_aware_select_vectorised
from strategy_ls_na_egbo import strategy_mo_egbo

from compose_sandbox import run_compose_in_sandbox, ComposeSandboxError
from da_coreg import fit_da_coreg_model
from fitness_common import to_allmax


def _fit_model(train_x, train_y, use_da_coreg: bool, tkwargs: dict):
    """Returns a botorch Model exposing .posterior() usable both directly
    by qLogNEHVI's acquisition function and for pool_mu/pool_sigma reads —
    either the standard ModelListGP(independent SingleTaskGPs) or a
    DA-COREG multi-task GP, per use_da_coreg."""
    if use_da_coreg:
        da_model = fit_da_coreg_model(train_x, train_y, tkwargs)
        return da_model._model  # raw botorch Model — valid for qLogNEHVI directly
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import SumMarginalLogLikelihood

    M = train_y.shape[1]
    models = [SingleTaskGP(train_x, train_y[:, j:j + 1], outcome_transform=Standardize(m=1))
              for j in range(M)]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll, max_attempts=1)
    return model


def strategy_ablation_cell(oracle, X_obs, Y_obs, bounds, batch_size, rng, budget,
                            use_da_coreg: bool = False, use_compose: bool = False,
                            compose_code: str = None, w_acq: float = 0.9,
                            w_nov: float = 0.1, sandbox_log_dir=None,
                            evo_candidates: int = 20, **kw):
    """
    One of the 2x2 ablation's four cells (see module docstring). The
    independent-GP + baseline-selection cell is NOT built here — it's
    strategy_mo_egbo_novelty, reused directly, since building it through
    this shared path would just be a slower reimplementation of code that
    already exists and is already validated everywhere else in this
    project.
    """
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import unnormalize
        from botorch.optim.optimize import optimize_acqf
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

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

        model = _fit_model(train_x, train_y, use_da_coreg, tkwargs)

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=batch_size,
                num_restarts=2, raw_samples=16,
                options={"maxiter": 20},
            )
        except Exception:
            qbo_x = torch.rand(batch_size, d, **tkwargs)

        pf_idx_np = pareto_front_of(Y_obs, directions=directions)
        seed_pool_idx = pf_idx_np if len(pf_idx_np) > 0 else np.arange(len(Y_obs))
        seed_x = train_x[seed_pool_idx].cpu().numpy()
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])
        seed_x = seed_x[:max(evo_candidates, 2)]

        try:
            ref_dirs = get_reference_directions("energy", M, evo_candidates,
                                                 seed=int(rng.integers(1e6)))
            pop_size = max(len(ref_dirs), evo_candidates, 2)
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x,
                          seed=int(rng.integers(1e6)))
            pm = PymooProblem(n_var=d, n_obj=M, n_constr=0,
                               xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            pop_size_actual = len(pop)
            f_idx = seed_pool_idx[:pop_size_actual] if \
                len(seed_pool_idx) >= pop_size_actual else seed_pool_idx
            f_vals = -Y_int[f_idx]
            if len(f_vals) >= pop_size_actual:
                pop.set("F", f_vals[:pop_size_actual])
            else:
                pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
                pop.set("F", np.vstack([f_vals, pad]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception:
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        candidates = torch.cat([qbo_x, ea_x], dim=0)
        candidates_n = candidates.cpu().numpy()
        step = len(X_obs)
        stagnant_batches = 0
        front_allmax = to_allmax(Y_obs.copy(), directions=directions)
        ref_point_allmax = ref_point.cpu().numpy()

        if use_compose:
            with warnings.catch_warnings(), torch.no_grad():
                warnings.simplefilter("ignore")
                post = model.posterior(candidates)
                pool_mu = post.mean.detach().cpu().numpy()
                pool_sigma = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()

            selected_idx = run_compose_in_sandbox(
                compose_code, candidates_n, pool_mu, pool_sigma, X_norm,
                front_allmax, ref_point_allmax, step, budget, step, stagnant_batches,
                batch_size, objective_names=oracle.objective_names(),
                log_dir=sandbox_log_dir,
            )
        else:
            with warnings.catch_warnings(), torch.no_grad():
                warnings.simplefilter("ignore")
                acq_vals = []
                for i in range(candidates.shape[0]):
                    try:
                        v = float(acq_fn(candidates[i].unsqueeze(0)).item())
                    except Exception:
                        v = float("-inf")
                    acq_vals.append(v)
                acq_vals = np.array(acq_vals)
            selected_idx = novelty_aware_select_vectorised(
                candidates_n, acq_vals, batch_size, w_acq=w_acq, w_nov=w_nov,
                X_obs_n=X_norm,
            )

        new_x_norm = candidates[selected_idx]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())

        return new_x_raw, {"use_da_coreg": use_da_coreg, "use_compose": use_compose,
                            "n_candidates": len(candidates)}

    except ComposeSandboxError:
        raise
    except Exception as e:
        warnings.warn(f"strategy_ablation_cell (da_coreg={use_da_coreg}, "
                       f"compose={use_compose}) failed ({e}), falling back to "
                       f"lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng)
        extra["ablation_cell_ok"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra
