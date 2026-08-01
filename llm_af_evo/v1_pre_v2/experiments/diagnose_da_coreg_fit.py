"""
diagnose_da_coreg_fit.py — isolates WHY DA-COREG lost on DTLZ2 (0/120
fallbacks confirmed, so it's not a crash) before treating "poor local
optimum from single-attempt MLE on a data-starved coregionalized fit" as
established rather than merely plausible. Two parts:

Part 1 — column-correspondence sanity check (synthetic, no real data at
all): fits fit_da_coreg_model on train_y with per-task offsets chosen to
be trivially distinguishable (0, 100, 200), then checks whether the
posterior mean at held-out points for output column j is close to task
j's offset. This directly tests the "different output shape/ordering
conventions" failure mode named as a classic silent-bug source (and one
this project has hit before — Part 2's hardcoded-objective-direction bug)
— independent of whether DTLZ2's fit is qualitatively good or bad.

Part 2 — held-out predictive quality on REAL data: reconstructs replicate
2's exact seed block (base_seed=42, replicate_idx=2 -> seed_offset=2042,
matching run_da_coreg_pilot.py's convention), fits both independent GPs
and DA-COREG on several campaigns' X_init/Y_init (10 points each — the
most data-starved point in any campaign, where the "poor local optimum"
story predicts the biggest gap), and compares held-out Gaussian negative
log-likelihood + RMSE against OTHER pool points' true (noiseless, exactly
known) DTLZ2 values. If DA-COREG's NLL is worse here, that's direct
evidence for the fit-quality story. If comparable, the losing decisions
in the actual campaigns are coming from somewhere else in the pipeline,
not surrogate fit quality.

Usage:
    python diagnose_da_coreg_fit.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from excipient_campaign_mo import make_shared_inits
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from da_coreg import fit_da_coreg_model

tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

# ── Part 1: column-correspondence sanity check ──────────────────────────────
print("=" * 70)
print("Part 1: does posterior column j actually correspond to task j?")
print("=" * 70)

rng = np.random.default_rng(0)
n, d, M = 15, 4, 3
train_x = torch.tensor(rng.random((n, d)), **tkwargs)
# Deliberately huge, trivially distinguishable per-task offsets — if the
# fitted model's posterior at held-out points doesn't cleanly separate
# into three well-separated bands near 0/100/200 in the RIGHT column
# order, that's a direct sign of a column/task-index wiring bug, not a
# subtle fit-quality issue (100 is orders of magnitude larger than any
# plausible GP noise/model-misspecification error).
offsets = np.array([0.0, 100.0, 200.0])
train_y_np = np.column_stack([
    offsets[j] + 0.01 * rng.normal(size=n) for j in range(M)
])
train_y = torch.tensor(train_y_np, **tkwargs)

da_model = fit_da_coreg_model(train_x, train_y, tkwargs)
test_x = torch.tensor(rng.random((5, d)), **tkwargs)
post = da_model.posterior(test_x)
mean = post.mean.detach().cpu().numpy()
print(f"Posterior mean shape: {mean.shape} (expect (5, {M}))")
print(f"Expected column means near: {offsets}")
print(f"Actual column means:        {mean.mean(axis=0)}")
col_order_ok = all(abs(mean[:, j].mean() - offsets[j]) < 10.0 for j in range(M))
print("=> COLUMN ORDER OK" if col_order_ok else
      "=> COLUMN ORDER MISMATCH — this is a wiring bug, not a fit-quality issue")

# ── Part 2: held-out predictive quality on real replicate-2 data ───────────
print("\n" + "=" * 70)
print("Part 2: held-out NLL/RMSE, independent GPs vs DA-COREG, on "
      "replicate 2's actual seed block")
print("=" * 70)

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
import warnings

oracle = DiscreteSyntheticMOOracle.build_dtlz2()
bounds = oracle.bounds()
lo, hi = bounds[:, 0], bounds[:, 1]
d = bounds.shape[0]
directions = oracle.objective_directions()  # all "min"
M = len(directions)

seed_offset = 42 + 2 * 1000  # matches run_da_coreg_pilot.py's replicate_idx=2 convention
n_campaigns_to_check = 5
inits = make_shared_inits(oracle, n_campaigns_to_check, 10, rng_seed=seed_offset)

# Held-out set: a fixed random sample of pool points, EXCLUDED from any
# campaign's init set, with known true (noiseless) Y — reused across all
# campaigns checked here for a consistent comparison.
held_out_rng = np.random.default_rng(seed_offset + 999)
held_out_idx = held_out_rng.choice(len(oracle), 100, replace=False)
X_held_raw = oracle._X_raw[held_out_idx]
Y_held_raw = oracle._Y_raw[held_out_idx].copy()
for j, direction in enumerate(directions):
    if direction == "min":
        Y_held_raw[:, j] = -Y_held_raw[:, j]  # all-maximise convention, same as every strategy fn
X_held_norm = (X_held_raw - lo) / (hi - lo + 1e-12)
X_held_t = torch.tensor(X_held_norm, **tkwargs)
Y_held_t = torch.tensor(Y_held_raw, **tkwargs)


def fit_indep(train_x, train_y):
    models = [SingleTaskGP(train_x, train_y[:, j:j + 1], outcome_transform=Standardize(m=1))
              for j in range(M)]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll, max_attempts=1)
    return model


def nll_and_rmse(model, X_test, Y_test):
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        post = model.posterior(X_test)
        mean = post.mean
        var = post.variance.clamp_min(1e-9)
    sq_err = (mean - Y_test) ** 2
    nll = 0.5 * (torch.log(2 * np.pi * var) + sq_err / var)
    rmse = torch.sqrt(sq_err.mean(dim=0))
    return float(nll.mean().item()), rmse.detach().cpu().numpy()


indep_nlls, da_nlls = [], []
for i, (X_init, Y_init) in enumerate(inits):
    X_norm = (X_init - lo) / (hi - lo + 1e-12)
    Y_int = Y_init.copy()
    for j, direction in enumerate(directions):
        if direction == "min":
            Y_int[:, j] = -Y_int[:, j]
    train_x = torch.tensor(X_norm, **tkwargs)
    train_y = torch.tensor(Y_int, **tkwargs)

    indep_model = fit_indep(train_x, train_y)
    da_model = fit_da_coreg_model(train_x, train_y, tkwargs)

    indep_nll, indep_rmse = nll_and_rmse(indep_model, X_held_t, Y_held_t)
    da_nll, da_rmse = nll_and_rmse(da_model, X_held_t, Y_held_t)
    indep_nlls.append(indep_nll)
    da_nlls.append(da_nll)

    print(f"\ncampaign {i} (10 init points):")
    print(f"  independent GPs: held-out NLL={indep_nll:+.3f}  RMSE per obj={indep_rmse}")
    print(f"  DA-COREG:        held-out NLL={da_nll:+.3f}  RMSE per obj={da_rmse}")

print(f"\n{'='*70}")
print(f"Mean held-out NLL — independent GPs: {np.mean(indep_nlls):+.3f}  "
      f"DA-COREG: {np.mean(da_nlls):+.3f}")
if np.mean(da_nlls) > np.mean(indep_nlls) + 0.5:
    print("=> DA-COREG's fit is meaningfully worse on held-out data — "
          "consistent with the poor-local-optimum/data-starved-fit story.")
elif np.mean(da_nlls) < np.mean(indep_nlls) - 0.5:
    print("=> DA-COREG's fit is actually BETTER on held-out data — the losing "
          "campaign decisions are NOT explained by surrogate fit quality; "
          "look elsewhere in the pipeline (e.g. how the acquisition function "
          "or downstream selection consumes the DA-COREG posterior).")
else:
    print("=> Held-out fit quality is comparable between the two surrogates — "
          "the losing campaign decisions are NOT explained by fit quality; "
          "look elsewhere in the pipeline.")

# ── Part 3: does qLogNEHVI's per-candidate acquisition evaluation silently
# fail more often against the DA-COREG (MultiTaskGP) posterior than against
# the independent-GP (ModelListGP) posterior? strategy_ablation_cell's own
# baseline-selection branch wraps EVERY acq_fn(candidates[i].unsqueeze(0))
# call in a bare try/except that converts any exception into -inf — if
# qLogNEHVI's caching optimisations (cache_root=True, prune_baseline=True,
# built and exercised mostly against ModelListGP elsewhere in this project)
# don't behave the same way against a MultiTaskGP's correlated joint
# posterior, this is exactly where it would show up: a good surrogate fit,
# silently degenerate (-inf-heavy) acquisition scores, and selection that's
# effectively close to random/novelty-only despite the model being fine. ──
print("\n" + "=" * 70)
print("Part 3: does the per-candidate acq_fn(candidate) call fail more often "
      "for DA-COREG than for independent GPs?")
print("=" * 70)

from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf

standard_bounds = torch.zeros(2, d, **tkwargs)
standard_bounds[1] = 1.0

for i, (X_init, Y_init) in enumerate(inits):
    X_norm = (X_init - lo) / (hi - lo + 1e-12)
    Y_int = Y_init.copy()
    for j, direction in enumerate(directions):
        if direction == "min":
            Y_int[:, j] = -Y_int[:, j]
    train_x = torch.tensor(X_norm, **tkwargs)
    train_y = torch.tensor(Y_int, **tkwargs)

    Y_all_int = oracle._Y_raw.copy()
    for j, direction in enumerate(directions):
        if direction == "min":
            Y_all_int[:, j] = -Y_all_int[:, j]
    ref_point = torch.tensor(
        Y_all_int.min(axis=0) - 0.1 * (Y_all_int.max(axis=0) - Y_all_int.min(axis=0) + 1e-9),
        **tkwargs)

    print(f"\ncampaign {i}:")
    for label, model in [("independent GPs", fit_indep(train_x, train_y)),
                          ("DA-COREG", fit_da_coreg_model(train_x, train_y, tkwargs))]:
        # DA-COREG's fit_da_coreg_model returns the DACoregModel wrapper;
        # qLogNEHVI needs the raw botorch Model, same as compose_strategies.
        raw_model = model._model if hasattr(model, "_model") else model
        acq_fn = qLogNoisyExpectedHypervolumeImprovement(
            model=raw_model, ref_point=ref_point, X_baseline=train_x,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
            prune_baseline=True, cache_root=True,
        )
        rng2 = np.random.default_rng(100 + i)
        candidates = torch.tensor(rng2.random((25, d)), **tkwargs)
        n_fail = 0
        vals = []
        first_error = None
        for k in range(candidates.shape[0]):
            try:
                v = float(acq_fn(candidates[k].unsqueeze(0)).item())
                vals.append(v)
            except Exception as e:
                n_fail += 1
                if first_error is None:
                    first_error = str(e)
        vals_arr = np.array(vals) if vals else np.array([float("nan")])
        print(f"  {label:<18} per-candidate acq_fn: {n_fail}/25 failed  "
              f"(succeeded: min={vals_arr.min():.4g} max={vals_arr.max():.4g} "
              f"std={vals_arr.std():.4g})")
        if first_error:
            print(f"    first failure: {first_error[:150]}")

        # Also check the qbo_x optimize_acqf path (q=batch_size, the OTHER
        # place strategy_ablation_cell silently falls back to torch.rand
        # on any exception).
        try:
            qbo_x, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=5, num_restarts=2,
                raw_samples=16, options={"maxiter": 20})
            print(f"    optimize_acqf: OK, qbo_x shape {tuple(qbo_x.shape)}")
        except Exception as e:
            print(f"    optimize_acqf: FAILED — {str(e)[:150]}")
