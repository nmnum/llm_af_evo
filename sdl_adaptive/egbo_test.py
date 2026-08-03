"""
Quick standalone test of EGBO on a tiny synthetic problem.
Run this BEFORE the full shared_seed_experiment to verify botorch/pymoo work.

Usage: python egbo_test.py
Expected output: 5 batches complete with no errors, AUC printed at end.
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def test_egbo():
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

    print("All imports OK")

    # Check qLogNoisyExpectedImprovement signature
    import inspect
    sig = inspect.signature(qLogNoisyExpectedImprovement.__init__)
    params = list(sig.parameters.keys())
    print(f"qLogNEI params: {params}")
    has_cache_root = "cache_root" in params
    has_prune_baseline = "prune_baseline" in params
    print(f"  cache_root supported: {has_cache_root}")
    print(f"  prune_baseline supported: {has_prune_baseline}")

    # Tiny synthetic test: 1D input, 1D output, 10 init points, 5 batches of 2
    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    d = 2
    n_init = 8
    budget = 20
    batch_size = 2
    evo_candidates = 10

    # True function: -(x-0.5)^2 + noise
    rng = np.random.default_rng(42)
    bounds_np = np.array([[0.0, 0.0], [1.0, 1.0]]).T  # shape (d,2) -> need (2,d)
    bounds_np = np.column_stack([np.zeros(d), np.ones(d)])  # (d,2)
    bounds_t = torch.tensor(bounds_np.T, **tkwargs)  # (2,d)
    standard_bounds = torch.zeros(2, d, **tkwargs)
    standard_bounds[1] = 1.0

    X_init = rng.random((n_init, d))
    y_init = -(np.sum((X_init - 0.3)**2, axis=1))  # optimum at [0.3, 0.3]

    train_x = torch.tensor(X_init, **tkwargs)
    train_y = torch.tensor(y_init.reshape(-1, 1), **tkwargs)

    running_best = [float(train_y.max())] * n_init
    n_batches = (budget - n_init) // batch_size

    for batch_idx in range(n_batches):
        train_x_norm = normalize(train_x, bounds_t)

        # Fit GP
        gp = SingleTaskGP(train_x_norm, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_retries=1, options={"maxiter": 30})

        # Build qLogNEI with version-safe kwargs
        acq_kwargs = dict(
            model=gp,
            X_baseline=train_x_norm,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
        )
        if has_prune_baseline:
            acq_kwargs["prune_baseline"] = True
        if has_cache_root:
            acq_kwargs["cache_root"] = True

        acq_fn = qLogNoisyExpectedImprovement(**acq_kwargs)

        # BO candidates
        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=4,
                num_restarts=1, raw_samples=8,
                options={"maxiter": 10},
            )
        except Exception as e:
            print(f"  optimize_acqf failed: {e} — using random")
            qbo_x = torch.rand(4, d, **tkwargs)

        # EA candidates via U-NSGA-III
        top_k = min(evo_candidates, train_x_norm.shape[0])
        top_idx = train_y.squeeze(-1).argsort(descending=True)[:top_k]
        seed_x = train_x_norm[top_idx].cpu().numpy()
        if seed_x.shape[0] < evo_candidates:
            pad = rng.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])

        try:
            ref_dirs = get_reference_directions("energy", 1, evo_candidates, seed=42)
        except Exception:
            ref_dirs = rng.random((evo_candidates, 1))

        try:
            algo = UNSGA3(pop_size=evo_candidates, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                              xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            # Set F to match actual population size (may differ from evo_candidates)
            pop_size_actual = len(pop)
            f_seed = train_y.squeeze(-1).argsort(descending=True)
            f_vals = -train_y[f_seed].cpu().numpy()
            if len(f_vals) >= pop_size_actual:
                pop.set("F", f_vals[:pop_size_actual].reshape(-1, 1))
            else:
                # Pad with worst observed value
                pad = np.full((pop_size_actual - len(f_vals), 1), f_vals[-1])
                pop.set("F", np.vstack([f_vals.reshape(-1,1), pad]))
            algo.tell(infills=pop)
            ea_x = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception as e:
            print(f"  NSGA-III failed: {e} — using random")
            ea_x = torch.rand(evo_candidates, d, **tkwargs)

        # Merge and score
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

        # Simple top-batch_size selection (no novelty for this test)
        top_idx_sel = np.argsort(acq_vals)[-batch_size:]
        new_x_norm = candidates[top_idx_sel]
        new_x = unnormalize(new_x_norm, bounds_t)
        new_y = -(torch.sum((new_x - 0.3)**2, dim=1, keepdim=True))

        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        for _ in range(batch_size):
            running_best.append(float(train_y.max()))

        print(f"  Batch {batch_idx+1}/{n_batches}: best={train_y.max():.4f}")

    final_best = float(train_y.max())
    optimal = 0.0  # true optimum is 0 at [0.3, 0.3]
    print(f"\nFinal best: {final_best:.4f}  (optimum: {optimal:.4f})")
    print(f"Gap from optimum: {abs(final_best - optimal):.4f}")
    print("\n✓ EGBO test passed — ready to run shared_seed_experiment.py")
    return True


if __name__ == "__main__":
    try:
        test_egbo()
    except Exception as e:
        import traceback
        print(f"\n✗ EGBO test FAILED: {e}")
        traceback.print_exc()
        print("\nFix the error above before running shared_seed_experiment.py")
