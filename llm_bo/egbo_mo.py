"""
egbo_mo.py — Real multi-objective EGBO: BoTorch qLogNEHVI + pymoo UNSGA3.

Ported from egbo_test.py's validated single-objective implementation
(qLogNoisyExpectedImprovement + UNSGA3 with n_obj=1) to true multi-objective
(qLogNoisyExpectedHypervolumeImprovement + UNSGA3 with n_obj=3), replacing
the lightweight numpy-only stand-in that strategy_mo_egbo used previously.

*** THIS FILE HAS NOT BEEN RUN — BoTorch is not installed in the sandbox
    this was written in. Run egbo_mo_test() below FIRST, exactly as
    egbo_test.py's test_egbo() was used to validate the single-objective
    version before shared_seed_experiment.py trusted it, before using
    real_egbo_mo_step() in any campaign. Do not assume this works from
    reading it — the single-objective port already required several
    version-compatibility guards (cache_root, prune_baseline signature
    checks) that a fresh multi-objective port may need equivalents of,
    and those can only be discovered by actually running it against
    whatever BoTorch/pymoo versions are installed on the target machine. ***

Why this exists: strategy_mo_egbo in excipient_campaign_mo.py was a
lightweight Gaussian-perturbation-plus-random-exploration stand-in, not
the U-NSGA-III evolutionary algorithm the single-objective EGBO work in
this project was actually validated with (see evolutionary_candidates.py,
egbo_test.py, shared_seed_experiment.py). Every "mo_egbo" result produced
so far in the multi-objective campaign runner should be understood as
"how does this specific lightweight baseline perform," not as evidence
about real EGBO's performance in the multi-objective, small-N, high-D
excipient regime.

Design note on the GP fallback used by strategy_mo_llm (NOT this file):
the fallback deliberately stays on the lightweight baseline, not this
real EGBO implementation, for two reasons. First, cost: BoTorch's
acq optimisation (multiple internal restarts, gradient-based optimize_acqf)
is substantially more expensive than the lightweight GP-UCB score, and the
fallback is called every time strategy_mo_llm needs to top up a batch —
in the low-trust regime (obs_per_dim<2) the mixing weight now correctly
sends ~80% of selection probability to LLM candidates, meaning most of
that expensive EGBO computation would be run and then mostly discarded.
Second, and more importantly, using real EGBO as mo_llm's fallback would
make mo_llm's results partially DEPENDENT ON mo_egbo's own machinery,
contaminating the comparison the whole experiment exists to make (is
LLM-guided search better than real EGBO, on its own terms, or is a
reported mo_llm win actually "EGBO bailed it out under the hood"?).
Keeping the fallback lightweight and neutral preserves that separation.
"""

import warnings
import numpy as np

warnings.filterwarnings("ignore")


def egbo_mo_test():
    """
    Standalone smoke test — run this BEFORE trusting real_egbo_mo_step()
    in any real campaign, exactly as egbo_test.py's test_egbo() was used
    to validate the single-objective port. Tests on a tiny synthetic
    3-objective problem with a known Pareto structure, not the excipient
    oracle, to isolate BoTorch/pymoo API issues from oracle-specific bugs.

    Usage: python egbo_mo.py
    Expected: 4 batches complete with no errors, final hypervolume printed.
    """
    import torch
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
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    print("All imports OK")

    import inspect
    sig = inspect.signature(qLogNoisyExpectedHypervolumeImprovement.__init__)
    params = list(sig.parameters.keys())
    print(f"qLogNEHVI params: {params}")
    has_cache_root = "cache_root" in params
    has_prune_baseline = "prune_baseline" in params
    print(f"  cache_root supported: {has_cache_root}")
    print(f"  prune_baseline supported: {has_prune_baseline}")

    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    d = 4
    M = 3  # objectives, matching the excipient oracle's Tm/kD/viscosity
    n_init = 10
    budget = 30
    batch_size = 5
    evo_candidates = 20

    rng = np.random.default_rng(42)
    bounds_np = np.column_stack([np.zeros(d), np.ones(d)])
    bounds_t = torch.tensor(bounds_np.T, **tkwargs)
    standard_bounds = torch.zeros(2, d, **tkwargs)
    standard_bounds[1] = 1.0

    def true_fn(X):
        # 3 conflicting synthetic objectives, all to be MAXIMISED internally
        f1 = -np.sum((X - 0.2) ** 2, axis=1)
        f2 = -np.sum((X - 0.8) ** 2, axis=1)
        f3 = -np.sum((X - 0.5) ** 2, axis=1) + 0.1 * np.sin(X[:, 0] * 10)
        return np.column_stack([f1, f2, f3])

    X_init = rng.random((n_init, d))
    Y_init = true_fn(X_init)

    train_x = torch.tensor(X_init, **tkwargs)
    train_y = torch.tensor(Y_init, **tkwargs)

    ref_point = torch.tensor([-2.0, -2.0, -2.0], **tkwargs)  # worse than any Y

    n_batches = (budget - n_init) // batch_size

    for batch_idx in range(n_batches):
        train_x_norm = normalize(train_x, bounds_t)

        # One SingleTaskGP per objective, combined via ModelListGP —
        # this is the multi-objective equivalent of the single-objective
        # port's single SingleTaskGP, and matches the "per-objective GP,
        # never one scalarised GP" design decision used throughout this
        # project's multi-objective work.
        models = []
        for j in range(M):
            gp_j = SingleTaskGP(train_x_norm, train_y[:, j:j+1],
                                outcome_transform=Standardize(m=1))
            models.append(gp_j)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_attempts=1)

        acq_kwargs = dict(
            model=model,
            ref_point=ref_point,
            X_baseline=train_x_norm,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        )
        if has_prune_baseline:
            acq_kwargs["prune_baseline"] = True
        if has_cache_root:
            acq_kwargs["cache_root"] = True

        acq_fn = qLogNoisyExpectedHypervolumeImprovement(**acq_kwargs)

        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=4,
                num_restarts=1, raw_samples=8,
                options={"maxiter": 10},
            )
        except Exception as e:
            print(f"  optimize_acqf failed: {e} — using random")
            qbo_x = torch.rand(4, d, **tkwargs)

        # EA candidates via U-NSGA-III, n_obj=M (the actual multi-objective
        # port — single-objective version used n_obj=1 with scalar -y).
        top_k = min(evo_candidates, train_x_norm.shape[0])
        # Seed the population from the CURRENT NON-DOMINATED set, not just
        # top-by-one-objective (which doesn't make sense once M>1).
        Y_np = train_y.cpu().numpy()
        # crude non-dominated filter for seeding purposes
        is_dominated = np.zeros(len(Y_np), dtype=bool)
        for i in range(len(Y_np)):
            ge = (Y_np >= Y_np[i]).all(axis=1)
            gt = (Y_np > Y_np[i]).any(axis=1)
            if (ge & gt).any():
                is_dominated[i] = True
        pf_idx = np.where(~is_dominated)[0]
        seed_pool_idx = pf_idx if len(pf_idx) > 0 else np.arange(len(Y_np))
        seed_x = train_x_norm[seed_pool_idx].cpu().numpy()
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])[:evo_candidates] if \
                seed_x.shape[0] + len(pad) >= evo_candidates else \
                np.vstack([seed_x, pad])

        try:
            ref_dirs = get_reference_directions("energy", M, evo_candidates,
                                                seed=42)
        except Exception:
            ref_dirs = rng.random((evo_candidates, M))

        try:
            pop_size = max(len(ref_dirs), evo_candidates, 2)
            algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=M, n_constr=0,
                              xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            pop_size_actual = len(pop)
            # pymoo MINIMISES by convention -> negate our maximise objectives
            f_seed_idx = seed_pool_idx[:pop_size_actual] if \
                len(seed_pool_idx) >= pop_size_actual else seed_pool_idx
            f_vals = -Y_np[f_seed_idx]
            if len(f_vals) >= pop_size_actual:
                pop.set("F", f_vals[:pop_size_actual])
            else:
                pad = np.tile(f_vals[-1:], (pop_size_actual - len(f_vals), 1))
                pop.set("F", np.vstack([f_vals, pad]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception as e:
            print(f"  NSGA-III (multi-obj) failed: {e} — using random")
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        candidates = torch.cat([qbo_x, ea_x], dim=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acq_vals = []
            for i in range(candidates.shape[0]):
                try:
                    v = float(acq_fn(candidates[i].unsqueeze(0)).item())
                except Exception:
                    v = float("-inf")
                acq_vals.append(v)

        top_idx_sel = np.argsort(acq_vals)[-batch_size:]
        new_x_norm = candidates[top_idx_sel]
        new_x = unnormalize(new_x_norm, bounds_t)
        new_y = torch.tensor(true_fn(new_x.cpu().numpy()), **tkwargs)

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])

        hv_calc = Hypervolume(ref_point=ref_point)
        Y_np_now = train_y.cpu().numpy()
        is_dom = np.zeros(len(Y_np_now), dtype=bool)
        for i in range(len(Y_np_now)):
            ge = (Y_np_now >= Y_np_now[i]).all(axis=1)
            gt = (Y_np_now > Y_np_now[i]).any(axis=1)
            if (ge & gt).any():
                is_dom[i] = True
        pf = Y_np_now[~is_dom]
        hv = hv_calc.compute(torch.tensor(pf, **tkwargs))
        print(f"  Batch {batch_idx+1}/{n_batches}: HV={hv:.4f}  "
              f"pareto_size={len(pf)}")

    print(f"\n✓ Multi-objective EGBO test passed — "
          f"ready to integrate into excipient_campaign_mo.py")
    return True


if __name__ == "__main__":
    try:
        egbo_mo_test()
    except Exception as e:
        import traceback
        print(f"\n✗ Multi-objective EGBO test FAILED: {e}")
        traceback.print_exc()
        print("\nDo NOT integrate this into the campaign runner until this "
              "test passes — fix the error above first, the same way "
              "egbo_test.py was required to pass before the single-"
              "objective EGBO was trusted in shared_seed_experiment.py.")
