"""
strategy_ls_na_egbo.py — LLM-Seeded Novelty-Aware EGBO strategy.

The recommended architecture from the plan:
  Stage 1: LLM warm-start generates the initial batch (called once, before
           any experiments, via llm_warmstart.py)
  Stage 2: Novelty-aware EGBO runs autonomously for the rest of the campaign
           (qLogNEHVI + U-NSGA-III evolutionary candidates + novelty-weighted
           batch selection from novelty_selection.py)

This file provides:
  - strategy_mo_egbo_novelty: EGBO with novelty-aware selection (no LLM)
  - strategy_mo_ls_na_egbo: LLM-seeded + novelty-aware EGBO (the full architecture)
  - run_ls_na_egbo_campaign: campaign runner that handles the two-stage flow

The strategy functions share the same interface as existing strategies in
excipient_campaign_mo.py (oracle, X_obs, Y_obs, bounds, batch_size, rng, **kw)
so they plug directly into the existing run_mo_campaign infrastructure.

Usage:
    from strategy_ls_na_egbo import strategy_mo_egbo_novelty, strategy_mo_ls_na_egbo

    # In excipient_campaign_mo.py's STRATEGIES dict:
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "mo_ls_na_egbo":   (strategy_mo_ls_na_egbo, {"prior_text": ..., "model": ...}),
"""

import warnings
import numpy as np
import torch

from excipient_campaign_mo import (
    strategy_mo_egbo, strategy_mo_egbo_real,
    pareto_front_of, OBJECTIVE_DIRECTIONS, OBJECTIVE_NAMES,
    _fit_gp_1d,
)
from novelty_selection import novelty_aware_select_vectorised
from llm_warmstart import llm_warmstart_init


# ── Novelty-aware EGBO (no LLM, just improved selection) ──────────────────────

def strategy_mo_egbo_novelty(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                              w_acq=0.9, w_nov=0.1, **kw):
    """
    EGBO with novelty-aware batch selection (Aqeeli et al. 2026).

    Same candidate generation as strategy_mo_egbo_real (qLogNEHVI +
    U-NSGA-III), but replaces greedy top-k selection with sequential
    novelty-weighted selection that prevents redundant candidates within
    a batch.
    """
    try:
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement)
        from botorch.models.gp_regression import SingleTaskGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import SumMarginalLogLikelihood
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from botorch.optim.optimize import optimize_acqf
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.core.termination import NoTermination
        from pymoo.util.ref_dirs import get_reference_directions

        tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        lo, hi = bounds[:, 0], bounds[:, 1]
        d = bounds.shape[0]
        M = Y_obs.shape[1]
        evo_candidates = 20
        # Oracle-provided directions, not the module-level OBJECTIVE_DIRECTIONS
        # constant — see excipient_campaign_mo.run_mo_campaign's identical
        # fix for why (correct only for the excipient oracle otherwise).
        directions = oracle.objective_directions()

        # Convert to "all maximise" convention
        Y_int = Y_obs.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_int[:, j] = -Y_int[:, j]

        X_norm = (X_obs - lo) / (hi - lo + 1e-12)
        train_x = torch.tensor(X_norm, **tkwargs)
        train_y = torch.tensor(Y_int, **tkwargs)
        standard_bounds = torch.zeros(2, d, **tkwargs)
        standard_bounds[1] = 1.0

        # Fixed reference point from full oracle pool
        Y_all_int = oracle._Y_raw.copy()
        for j, direction in enumerate(directions):
            if direction == "min":
                Y_all_int[:, j] = -Y_all_int[:, j]
        ref_point = torch.tensor(
            Y_all_int.min(axis=0) -
            0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
            **tkwargs)

        # Fit per-objective GPs
        models = [SingleTaskGP(train_x, train_y[:, j:j+1],
                               outcome_transform=Standardize(m=1))
                 for j in range(M)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        # qLogNEHVI acquisition
        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )

        # Acquisition-optimised candidates
        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=batch_size,
                num_restarts=2, raw_samples=16,
                options={"maxiter": 20},
            )
        except Exception:
            qbo_x = torch.rand(batch_size, d, **tkwargs)

        # Evolutionary candidates (U-NSGA-III)
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

        # Merge candidate pools
        candidates = torch.cat([qbo_x, ea_x], dim=0)

        # Score all candidates with acquisition function in a single batched
        # call (candidates.unsqueeze(1) -> shape (N, 1, d), i.e. N independent
        # q=1 evaluations), instead of looping one candidate at a time — the
        # per-candidate loop was the dominant cost of this strategy, since it
        # repeated the MC-sampling forward pass N times instead of once.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                acq_vals = acq_fn(candidates.unsqueeze(1)).detach().cpu().numpy()
            except Exception:
                # Fall back to per-candidate scoring if batched eval fails
                # for any reason (e.g. a single degenerate candidate).
                acq_vals = []
                for i in range(candidates.shape[0]):
                    try:
                        v = float(acq_fn(candidates[i].unsqueeze(0)).item())
                    except Exception:
                        v = float("-inf")
                    acq_vals.append(v)
                acq_vals = np.array(acq_vals)
        candidates_n = candidates.cpu().numpy()

        # ── KEY DIFFERENCE: novelty-aware selection instead of greedy top-k ──
        X_obs_n = X_norm  # already normalised
        selected_idx = novelty_aware_select_vectorised(
            candidates_n, acq_vals, batch_size,
            w_acq=w_acq, w_nov=w_nov,
            X_obs_n=X_obs_n,
        )

        new_x_norm = candidates[selected_idx]
        new_x_raw = (unnormalize(new_x_norm, torch.tensor(
            np.column_stack([lo, hi]).T, **tkwargs)).cpu().numpy())

        # Pool-level GP posterior mean, in the same "all-maximise" internal
        # convention as train_y above (flip back with OBJECTIVE_DIRECTIONS
        # when consuming downstream). This is the piece llm_af_evo's
        # counterfactual diagnostic needs and that previously never left
        # this function: without it, "what did the strategy predict for
        # every pool candidate" is unrecoverable after this call returns,
        # since only the batch_size selected points ever got returned.
        # Additive only — does not affect selection or fitted models above.
        # pool_pred_sigma (posterior std, same all-maximise/real-units
        # convention as pool_pred_mu — BoTorch's Standardize outcome
        # transform untransforms both mean and variance back to Y_int's
        # scale by default) is what llm_af_evo's exploration-credit fitness
        # needs; without it only the GP mean was recoverable, which is
        # enough for the pure-exploitation counterfactual but not for
        # scoring an actual exploration term.
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            try:
                post = model.posterior(candidates)
                pool_pred_mu = post.mean.detach().cpu().numpy()
                pool_pred_sigma = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy()
            except Exception:
                pool_pred_mu = np.full((len(candidates), M), np.nan)
                pool_pred_sigma = np.full((len(candidates), M), np.nan)

        return new_x_raw, {
            "real_egbo": True,
            "novelty_select": True,
            "w_acq": w_acq,
            "w_nov": w_nov,
            "n_candidates": len(candidates),
            "acq_range": [float(acq_vals.min()), float(acq_vals.max())],
            # Full candidate pool this batch considered, for counterfactual
            # re-scoring: normalised x, GP-predicted objective vector
            # (all-maximise convention), qLogNEHVI acquisition value, and
            # which pool indices were actually selected.
            "pool_x_norm": candidates_n.tolist(),
            "pool_pred_mu": pool_pred_mu.tolist(),
            "pool_pred_sigma": pool_pred_sigma.tolist(),
            "pool_acq_vals": [float(v) for v in acq_vals],
            "pool_selected_idx": [int(i) for i in selected_idx],
        }

    except Exception as e:
        warnings.warn(f"strategy_mo_egbo_novelty failed ({e}), "
                      f"falling back to lightweight strategy_mo_egbo.")
        cands, extra = strategy_mo_egbo(oracle, X_obs, Y_obs, bounds,
                                        batch_size, rng, **kw)
        extra["novelty_select"] = False
        extra["fallback_reason"] = str(e)
        return cands, extra


# ── LLM-seeded novelty-aware EGBO (the full architecture) ─────────────────────

def strategy_mo_ls_na_egbo(oracle, X_obs, Y_obs, bounds, batch_size, rng,
                            w_acq=0.9, w_nov=0.1, **kw):
    """
    LLM-Seeded Novelty-Aware EGBO.

    During the campaign (after warm-start), this is identical to
    strategy_mo_egbo_novelty. The LLM warm-start happens BEFORE the campaign
    loop starts, in run_ls_na_egbo_campaign — this function is only called
    for batches after the initial LLM-selected points.

    The LLM's contribution is entirely in the initial design: better starting
    points → better GP lengthscale estimates → better EGBO acquisition from
    the first batch onward.
    """
    return strategy_mo_egbo_novelty(
        oracle, X_obs, Y_obs, bounds, batch_size, rng,
        w_acq=w_acq, w_nov=w_nov, **kw,
    )


# ── Campaign runner with LLM warm-start ────────────────────────────────────────

def run_ls_na_egbo_campaign(
    disc_oracle,
    budget: int,
    strategy_fn,
    strategy_kwargs: dict,
    batch_size: int = 5,
    seed: int = 0,
    n_init: int = 10,
    n_propose: int = 25,
    protein: str = "mAb_aggregation",
    prior_level: str = "L1",
    model: str = "qwen3:32b",
    mock_llm: bool = False,
):
    """
    Run a campaign with LLM warm-start + novelty-aware EGBO.

    Stage 1: LLM generates n_propose formulations, diversity-select n_init,
             query oracle for initial objectives.
    Stage 2: Run novelty-aware EGBO for the remaining budget.

    This bypasses the standard make_shared_inits + run_mo_campaign flow
    because the LLM warm-start produces DIFFERENT initial points per seed
    (the LLM is called with different random seeds for diversity), unlike
    the shared-init design where all conditions see the same starting points.

    For fair comparison against random-init conditions, the random-init
    conditions should use the same n_init and budget.
    """
    from excipient_campaign_mo import (
        snap_query_mo, pareto_front_of, run_mo_campaign,
        make_shared_inits,
    )

    bounds = disc_oracle.bounds()
    rng = np.random.default_rng(seed)

    # ── Stage 1: LLM warm-start ──
    X_init, Y_init, forms, warmstart_meta = llm_warmstart_init(
        disc_oracle, protein=protein, prior_level=prior_level,
        n_propose=n_propose, n_select=n_init,
        model=model, mock=mock_llm, seed=seed,
    )

    # Register init points as queried
    disc_oracle._queried = set()
    X_all_s = disc_oracle._scaler.transform(disc_oracle._X_raw)
    for row in disc_oracle._scaler.transform(X_init):
        idx = int(np.argmin(np.linalg.norm(X_all_s - row, axis=1)))
        disc_oracle._queried.add(idx)

    # ── Stage 2: Run novelty-aware EGBO for remaining budget ──
    # Use the existing run_mo_campaign with the LLM-selected init points
    result = run_mo_campaign(
        disc_oracle, X_init, Y_init, budget, strategy_fn, strategy_kwargs,
        batch_size=batch_size, seed=seed,
    )

    # Add warm-start metadata
    result["warmstart_forms"] = forms
    result["prior_level"] = prior_level
    result["protein"] = protein
    result["n_init_llm"] = n_init
    result["llm_fallback_used"] = warmstart_meta["llm_fallback_used"]
    result["llm_retry_count"] = warmstart_meta["llm_retry_count"]

    return result
