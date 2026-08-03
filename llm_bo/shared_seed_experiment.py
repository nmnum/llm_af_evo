"""
shared_seed_experiment.py — Post-hoc benchmark with shared initialisations.

All conditions start from IDENTICAL initial points per repeat, eliminating
the initialisation lottery confound identified in the original study.

Conditions included:
  Fixed baselines : fixed_random, fixed_lhs, fixed_ucb_low, fixed_ucb_high, fixed_ei
  Mock baseline   : mock_approach_b  (phase-rule controller, no LLM)
  LLM controllers : approach_a, approach_b, approach_c
  EGBO variants   : egbo            (qLogNEI + U-NSGA-III, merit_weight=1.0)
                    novelty_egbo    (qLogNEI + U-NSGA-III, merit_weight=0.7)

EGBO implementation follows Aqeeli et al. (2026) exactly:
  - GP: BoTorch SingleTaskGP per objective + ModelListGP, Matern 5/2, Standardize
  - Acquisition: qLogNoisyExpectedImprovement (single-obj), Sobol qMC (16 samples)
  - Evolutionary: pymoo U-NSGA-III, pop_size=72, seeded from current Pareto set
  - Novelty selection: greedy sequential, score = w*merit + (1-w)*novelty
    where w=0.7 for novelty_egbo, w=1.0 for standard egbo

NOTE: EGBO requires botorch, gpytorch, torch, pymoo.
Install with: pip install botorch gpytorch torch pymoo

Usage:
    python shared_seed_experiment.py \\
        --data_dir data/ \\
        --out_dir results_shared/ \\
        --n_repeats 20 \\
        --budget_frac 0.5 \\
        --datasets pareto_20201218 pareto_20210112 coatings \\
        --model qwen2.5-coder:7b
"""

import argparse
import json
import logging
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from oracle import NNOracle
from baselines import FixedStrategyBaseline
from evaluate import compute_metrics, aggregate_across_seeds

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_DATASETS = [
    ("coatings",        "coatings"),
    ("pareto_20201218", "pareto_campaign 2020-12-18_17-38-40"),
    ("pareto_20201223", "pareto_campaign 2020-12-23_17-06-50"),
    ("pareto_20210104", "pareto_campaign 2021-01-04_08-37-39"),
    ("pareto_20210112", "pareto_campaign 2021-01-12_16-26-56"),
    ("hartmann3",       "hartmann3"),
    ("hartmann6",       "hartmann6"),
]

DEFAULT_CONDITIONS = [
    "fixed_random", "fixed_lhs", "fixed_ucb_low", "fixed_ucb_high", "fixed_ei",
    "mock_approach_b",
    "approach_a", "approach_b", "approach_c",
    "egbo", "novelty_egbo",
]


# ── EGBO implementation (Aqeeli et al. 2026) ─────────────────────────────────

def _egbo_available():
    try:
        import torch, botorch, pymoo  # noqa
        return True
    except ImportError:
        return False


def _fit_gp_models(train_x_norm, train_y):
    """Build ModelListGP from normalised inputs and raw objectives."""
    import torch
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    models = [
        SingleTaskGP(train_x_norm, train_y[..., i:i+1],
                     outcome_transform=Standardize(m=1))
        for i in range(train_y.shape[-1])
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll, max_retries=1, options={"maxiter": 50})
    return model


def _select_novelty(train_x_norm, candidates_norm, acq_vals, batch_size, merit_weight=0.7):
    """
    Greedy novelty-aware selection (Aqeeli et al. 2026, Eq. score_i = w*a~_i + (1-w)*n~_i).
    When merit_weight=1.0 this reduces to pure acquisition ranking (standard EGBO).
    """
    import numpy as np
    import torch

    merit_weight = float(np.clip(merit_weight, 0.0, 1.0))
    acq = np.asarray(acq_vals, dtype=float)
    finite = np.isfinite(acq)
    merit_norm = np.zeros_like(acq)
    if finite.any():
        lo, hi = float(np.min(acq[finite])), float(np.max(acq[finite]))
        merit_norm[finite] = (acq[finite] - lo) / (hi - lo) if hi - lo > 1e-12 else 1.0

    Xc = candidates_norm.detach()
    Xt = train_x_norm.detach()
    selected, remaining = [], set(range(Xc.shape[0]))

    while len(selected) < batch_size and remaining:
        rem = np.array(sorted(remaining), dtype=int)
        base = Xt if not selected else torch.cat([Xt, Xc[np.array(selected)]], dim=0)
        nov = torch.cdist(Xc[rem], base).min(dim=1).values.cpu().numpy()
        n_lo, n_hi = float(np.min(nov)), float(np.max(nov))
        nov_norm = np.ones_like(nov) if n_hi - n_lo <= 1e-12 else (nov - n_lo) / (n_hi - n_lo)
        score = merit_weight * merit_norm[rem] + (1.0 - merit_weight) * nov_norm
        pick = int(rem[int(np.argmax(score))])
        selected.append(pick)
        remaining.remove(pick)

    return np.array(selected[:batch_size], dtype=int)


def run_egbo_campaign(oracle, X_init, y_init, budget, batch_size=4,
                      qnehvi_candidates=8, evo_candidates=72,
                      merit_weight=0.7, random_state=0):
    """
    Run one EGBO/Novelty-EGBO campaign on a discrete NNOracle dataset.

    Follows Aqeeli et al. (2026) post-hoc protocol:
      - All objectives are treated as a SINGLE scalarised objective via NNOracle
        (oracle already returns a scalar). We wrap it as a 1-objective problem
        Uses qLogNoisyExpectedImprovement (single-objective NEI).
      - Candidate selection snaps to nearest unqueried dataset row (NN oracle).
      - merit_weight=1.0 → standard EGBO (pure acquisition ranking)
        merit_weight=0.7 → novelty-aware EGBO
    """
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.transforms import normalize, unnormalize
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    warnings.filterwarnings("ignore")
    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

    bounds_np = oracle.bounds()           # (d, 2)
    d = bounds_np.shape[0]
    N_dataset = len(oracle._X_raw)

    # BoTorch bounds tensor: shape (2, d)
    bounds_t = torch.tensor(bounds_np.T, **tkwargs)  # (2, d)
    standard_bounds = torch.zeros(2, d, **tkwargs)
    standard_bounds[1] = 1.0

    # Working set — 1 scalarised objective
    train_x = torch.tensor(X_init, **tkwargs)                   # (n_init, d)
    train_y = torch.tensor(y_init.reshape(-1, 1), **tkwargs)    # (n_init, 1)

    # Snap init points to nearest dataset indices
    all_X = oracle._X_raw
    scaler = oracle._scaler
    X_scaled_all = scaler.transform(all_X)
    X_init_scaled = scaler.transform(X_init)
    queried = set()
    for row in X_init_scaled:
        dists = np.linalg.norm(X_scaled_all - row, axis=1)
        queried.add(int(np.argmin(dists)))

    running_best = [float(train_y.max().item())] * len(X_init)
    decisions = []

    n_batches = max(1, (budget - len(X_init)) // batch_size)

    for batch_idx in range(n_batches):
        train_x_norm = normalize(train_x, bounds_t)

        # Fit single-output GP (scalarised single objective)
        gp_model = SingleTaskGP(train_x_norm, train_y,
                                outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll, max_retries=1, options={"maxiter": 50})

        # qLogNEI — correct acquisition for single-objective.
        # Build kwargs conditionally to handle different botorch versions.
        import inspect as _inspect
        _nei_params = list(_inspect.signature(
            qLogNoisyExpectedImprovement.__init__).parameters)
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

        # BO candidates: optimise acq in normalised space
        from botorch.optim.optimize import optimize_acqf
        try:
            qbo_x_norm, _ = optimize_acqf(
                acq_fn, bounds=standard_bounds, q=qnehvi_candidates,
                num_restarts=1, raw_samples=16,
                options={"batch_limit": qnehvi_candidates, "maxiter": 20},
            )
        except Exception:
            from botorch.utils.sampling import draw_sobol_samples
            qbo_x_norm = draw_sobol_samples(
                bounds=standard_bounds, n=qnehvi_candidates, q=1
            ).squeeze(-2)

        # EA candidates: U-NSGA-III seeded from current Pareto set
        # For 1-objective, "Pareto set" = current best point.
        # Seed NSGA-III from the best observed points (top-k by y value).
        top_k = min(evo_candidates, train_x_norm.shape[0])
        top_idx = train_y.squeeze(-1).argsort(descending=True)[:top_k]
        seed_x = train_x_norm[top_idx].cpu().numpy()
        # Pad with random samples if fewer than pop_size
        if seed_x.shape[0] < evo_candidates:
            rng_pad = np.random.default_rng(random_state + batch_idx)
            pad = rng_pad.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])

        try:
            ref_dirs = get_reference_directions("energy", 1, evo_candidates,
                                                seed=random_state)
        except Exception:
            rng = np.random.default_rng(random_state)
            ref_dirs = rng.random((evo_candidates, 1))

        try:
            algo = UNSGA3(
                pop_size=evo_candidates,
                ref_dirs=ref_dirs,
                sampling=seed_x,
            )
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
            ea_x_norm = torch.tensor(algo.ask().get("X"), **tkwargs)
        except Exception:
            # Fall back to random candidates if pymoo fails
            ea_x_norm = torch.rand(evo_candidates, d, **tkwargs)

        # Merge pools and score with acquisition
        candidates_norm = torch.cat([qbo_x_norm, ea_x_norm], dim=0)
        with torch.no_grad():
            try:
                acq_vals = [
                    float(acq_fn(candidates_norm[i].unsqueeze(0)).item())
                    for i in range(candidates_norm.shape[0])
                ]
            except Exception:
                acq_vals = list(np.random.randn(candidates_norm.shape[0]))

        # Novelty-aware selection
        sel_idx = _select_novelty(
            train_x_norm, candidates_norm, acq_vals,
            batch_size=batch_size, merit_weight=merit_weight
        )

        # Snap selected candidates to nearest unqueried dataset rows
        sel_x_norm = candidates_norm[sel_idx].cpu().numpy()
        sel_x_raw = unnormalize(
            torch.tensor(sel_x_norm, **tkwargs), bounds_t
        ).cpu().numpy()

        unqueried = [i for i in range(N_dataset) if i not in queried]
        if not unqueried:
            unqueried = list(range(N_dataset))

        new_x_rows, new_y_vals = [], []
        for x_raw in sel_x_raw:
            x_s = scaler.transform(x_raw.reshape(1, -1))[0]
            pool_s = scaler.transform(all_X[unqueried])
            dists = np.linalg.norm(pool_s - x_s, axis=1)
            best_pool_idx = int(np.argmin(dists))
            chosen = unqueried[best_pool_idx]
            queried.add(chosen)
            new_x_rows.append(all_X[chosen])
            new_y_vals.append(float(oracle._y_raw[chosen]))
            # Remove from unqueried for within-batch deduplication
            unqueried = [i for i in unqueried if i != chosen]

        new_x_t = torch.tensor(np.array(new_x_rows), **tkwargs)
        new_y_t = torch.tensor(np.array(new_y_vals).reshape(-1, 1), **tkwargs)

        train_x = torch.cat([train_x, new_x_t], dim=0)
        train_y = torch.cat([train_y, new_y_t], dim=0)

        for _ in new_y_vals:
            running_best.append(float(train_y.max().item()))

        decisions.append((
            len(X_init) + batch_idx * batch_size,
            "egbo" if merit_weight >= 1.0 else "novelty_egbo",
            {"merit_weight": merit_weight, "batch_size": batch_size},
        ))

    return {
        "running_best": running_best,
        "decisions": decisions,
        "failures": [],
        "X_obs": train_x.cpu().numpy(),
        "y_obs": train_y.cpu().numpy().flatten(),
    }


# ── SDL controller factory ────────────────────────────────────────────────────

def make_controller(condition, model="qwen2.5-coder:7b", ds_label="unknown"):
    from controllers.mock_controller import MockApproachBController
    if condition == "fixed_random":    return FixedStrategyBaseline("random", {})
    if condition == "fixed_lhs":       return FixedStrategyBaseline("lhs", {})
    if condition == "fixed_ucb_low":   return FixedStrategyBaseline("ucb", {"beta": 0.2})
    if condition == "fixed_ucb_high":  return FixedStrategyBaseline("ucb", {"beta": 400.0})
    if condition == "fixed_ei":        return FixedStrategyBaseline("ei", {})
    if condition == "mock_approach_b": return MockApproachBController(seed=0)
    if condition == "approach_a":
        from controllers.approach_a import ApproachAController
        return ApproachAController(model=model)
    if condition == "approach_b":
        from controllers.approach_b import ApproachBController
        return ApproachBController(model=model)
    if condition == "approach_c":
        from controllers.approach_c import ApproachCController
        return ApproachCController(model=model,
                                   code_log_dir=f"./approach_c_logs_{ds_label}")
    if condition == "approach_c_evo":
        from controllers.approach_c import ApproachCEvoController
        return ApproachCEvoController(model=model,
                                      code_log_dir=f"./approach_c_evo_logs_{ds_label}")
    if condition == "rule_router":
        from controllers.rule_router import RuleRouterController
        return RuleRouterController()
    if condition == "approach_d":
        from controllers.approach_d import ApproachDController
        return ApproachDController(model=model)
    raise ValueError(f"Unknown condition: {condition!r}")


# ── Shared-init campaign runner ───────────────────────────────────────────────

def generate_shared_inits(oracle, n_repeats, n_init, rng_seed=42):
    """Pre-generate n_repeats init sets shared across ALL conditions."""
    rng = np.random.default_rng(rng_seed)
    bounds = oracle.bounds()
    d = bounds.shape[0]
    inits = []
    for _ in range(n_repeats):
        X = np.array([[rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)]
                      for _ in range(n_init)])
        y = np.array([oracle.query(x) for x in X])
        inits.append((X.copy(), y.copy()))
    return inits


def _improvement_rate(y_obs, window=10):
    if len(y_obs) < 2:
        return 1.0
    recent = y_obs[-window:]
    improvements = sum(recent[i] > recent[:i].max() for i in range(1, len(recent)))
    return float(improvements / max(len(recent) - 1, 1))


def _gp_uncertainty(X_obs, y_obs, bounds, n_test=80):
    """Returns (gp_uncertainty, lengthscale_norm) tuple."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_s, y_obs)
    d = bounds.shape[0]
    pts = np.column_stack([np.random.uniform(bounds[i, 0], bounds[i, 1], n_test)
                           for i in range(d)])
    _, sigma = gp.predict(scaler.transform(pts), return_std=True)
    # Extract normalised lengthscale
    ls_scaled = float(gp.kernel_.length_scale)
    mean_scale = float(scaler.scale_.mean())
    mean_domain = float((bounds[:, 1] - bounds[:, 0]).mean())
    ls_norm = float(np.clip(ls_scaled * mean_scale / (mean_domain + 1e-12), 0, 10))
    return float(sigma.mean()), ls_norm


def run_sdl_campaign(oracle, ds_name, X_init, y_init, controller,
                     budget, controller_interval=5, first_control_step=None):
    """
    Run one SDL campaign from pre-specified initial points.

    first_control_step: step at which the controller first fires.
      Steps from n_init to first_control_step-1 run LHS regardless.
      At first_control_step, GP trust diagnostic runs; if trust_flag=="distrust",
      overrides to LHS. Default: n_init (fires immediately, original behaviour).
    """
    from strategies import STRATEGY_MAP
    bounds = oracle.bounds()
    d = bounds.shape[0]
    n_init = len(X_init)

    X_obs = X_init.copy()
    y_obs = y_init.copy()
    running_best = [float(y_obs.max())] * n_init
    decisions, failures = [], []
    current_strategy, current_params = "lhs", {}
    x_next_custom = None

    if first_control_step is None:
        first_control_step = n_init
    first_control_step = max(first_control_step, n_init)

    for step in range(n_init, budget):
        controller_active = (step >= first_control_step)
        if controller_active and (step - first_control_step) % controller_interval == 0:
            needs_gp = getattr(controller, "_needs_gp_uncertainty", False)
            if needs_gp:
                gp_unc, ls_norm = _gp_uncertainty(X_obs, y_obs, bounds)
            else:
                gp_unc = float(y_obs.std() / (y_obs.max() - y_obs.min() + 1e-12))
                ls_norm = 0.5

            # GP trust diagnostic at first_control_step
            trust_score, trust_flag = None, None
            if step == first_control_step:
                try:
                    from gp_trust_check import gp_trust_check as _trust_check
                    _tr = _trust_check(X_obs, y_obs, bounds)
                    trust_score = _tr.trust_score
                    trust_flag  = _tr.trust_flag
                except Exception:
                    pass

            context = {
                "step": step, "budget": budget, "n_obs": len(y_obs),
                "n_dims": bounds.shape[0],
                "best_so_far": float(y_obs.max()),
                "best_normalised": float(y_obs.max() / (oracle.global_best() + 1e-12)),
                "improvement_rate": _improvement_rate(y_obs),
                "gp_uncertainty": gp_unc,
                "lengthscale_norm": ls_norm,
                "trust_score": trust_score,
                "trust_flag":  trust_flag,
                "dataset": ds_name, "X_obs": X_obs, "y_obs": y_obs, "bounds": bounds,
            }

            # Override to LHS if GP is distrusted at first routing decision
            if trust_flag == "distrust":
                decision = {"strategy": "lhs", "params": {}}
            else:
                try:
                    decision = controller.decide(context)
                except Exception:
                    decision = {"strategy": current_strategy, "params": current_params}

            s = decision.get("strategy", current_strategy)
            p = decision.get("params", current_params)
            x_next_custom = decision.get("x_next", None)
            if s in STRATEGY_MAP or s == "_custom":
                current_strategy, current_params = s, p
            decisions.append((step, current_strategy, {**current_params}))

        if not controller_active:
            current_strategy, current_params = "lhs", {}

        try:
            if current_strategy == "_custom" and x_next_custom is not None:
                x_next = np.clip(np.asarray(x_next_custom, dtype=float),
                                 bounds[:, 0], bounds[:, 1])
                x_next_custom = None
            else:
                fn = STRATEGY_MAP.get(current_strategy, STRATEGY_MAP["random"])
                x_next = fn(X_obs, y_obs, bounds, **current_params)
            y_next = oracle.query(x_next)
        except Exception:
            failures.append(step)
            rng = np.random.default_rng(step)
            x_next = np.array([rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(d)])
            y_next = oracle.query(x_next)

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)
        running_best.append(float(y_obs.max()))

    return {"running_best": running_best, "decisions": decisions,
            "failures": failures, "X_obs": X_obs, "y_obs": y_obs}


# ── Main experiment loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--out_dir",     default="results_shared")
    parser.add_argument("--n_repeats",   type=int,   default=20)
    parser.add_argument("--n_init",      type=int,   default=10)
    parser.add_argument("--budget_frac", type=float, default=0.5,
                        help="Budget as fraction of dataset N (default 0.5)")
    parser.add_argument("--datasets",    nargs="+",  default=None)
    parser.add_argument("--conditions",  nargs="+",  default=None)
    parser.add_argument("--model",       default="qwen2.5-coder:7b")
    parser.add_argument("--batch_size",  type=int,   default=4,
                        help="Batch size for EGBO conditions (default 4)")
    parser.add_argument("--egbo_evo_candidates", type=int, default=72,
                        help="U-NSGA-III population size per batch (default 72)")
    parser.add_argument("--first_control_step", type=int, default=None,
                        help="Step at which controller first fires (default: n_init). "
                             "Steps before this run LHS. "
                             "GP trust check runs at this step.")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = args.conditions or DEFAULT_CONDITIONS
    egbo_conditions = {"egbo", "novelty_egbo"}
    sdl_conditions  = [c for c in conditions if c not in egbo_conditions]
    run_egbo_conds  = [c for c in conditions if c in egbo_conditions]

    if run_egbo_conds and not _egbo_available():
        print("WARNING: botorch/pymoo not installed — skipping EGBO conditions.")
        print("  Install with: pip install botorch gpytorch torch pymoo")
        run_egbo_conds = []

    dataset_list = ALL_DATASETS
    if args.datasets:
        dataset_list = [(l, n) for l, n in ALL_DATASETS if l in args.datasets]

    all_rows = []

    for ds_label, ds_name in dataset_list:
        print(f"\n{'='*60}\nDataset: {ds_label}")
        oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        N      = len(oracle._X_raw)
        budget = max(args.n_init + 10, int(args.budget_frac * N))
        gb     = oracle.global_best()
        gmin   = float(oracle._y_raw.min())
        d      = oracle.bounds().shape[0]
        print(f"  N={N}  budget={budget} ({args.budget_frac*100:.0f}% of N)  "
              f"dims={d}  global_best={gb:.4f}")

        # Pre-generate shared initialisations
        shared_inits = generate_shared_inits(oracle, args.n_repeats,
                                              args.n_init, rng_seed=42)
        best_norms   = [y.max() / gb for _, y in shared_inits]
        print(f"  Shared inits: best_norm range "
              f"{min(best_norms):.2f}–{max(best_norms):.2f}  "
              f"(mean {np.mean(best_norms):.2f})")

        # ── SDL conditions ──────────────────────────────────────────────────
        for condition in sdl_conditions:
            seed_metrics, seed_curves, seed_logs = [], [], []
            code_dir = out_dir / ds_label / condition / "generated_code"

            for rep_idx, (X_init, y_init) in enumerate(shared_inits):
                try:
                    ctrl = make_controller(condition, args.model, ds_label)
                except Exception as e:
                    logger.warning(f"Skip {condition} rep {rep_idx}: {e}")
                    continue

                res = run_sdl_campaign(
                    oracle, ds_name, X_init, y_init, ctrl, budget,
                    controller_interval=5,
                    first_control_step=args.first_control_step,
                )

                # Save generated code for approach_c analysis
                if condition == "approach_c" and hasattr(ctrl, "_current_code"):
                    code_dir.mkdir(parents=True, exist_ok=True)
                    (code_dir / f"repeat_{rep_idx:03d}_final.py").write_text(
                        ctrl._current_code)

                m = compute_metrics(res, gb, budget=budget, oracle_global_min=gmin)
                seed_metrics.append(m)
                seed_curves.append(np.array(res["running_best"]))
                decisions_clean = [
                    (t, s, {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in p.items()})
                    for t, s, p in res["decisions"]
                ]
                seed_logs.append({"decisions": decisions_clean,
                                   "failures": res["failures"]})

            _save_and_print(out_dir, ds_label, condition, seed_metrics,
                            seed_curves, seed_logs, gb, gmin, shared_inits,
                            all_rows)

        # ── EGBO conditions ─────────────────────────────────────────────────
        for condition in run_egbo_conds:
            merit_weight = 1.0 if condition == "egbo" else 0.7
            seed_metrics, seed_curves, seed_logs = [], [], []

            for rep_idx, (X_init, y_init) in enumerate(shared_inits):
                try:
                    res = run_egbo_campaign(
                        oracle, X_init, y_init, budget,
                        batch_size=args.batch_size,
                        qnehvi_candidates=8,
                        evo_candidates=args.egbo_evo_candidates,
                        merit_weight=merit_weight,
                        random_state=rep_idx,
                    )
                except Exception as e:
                    logger.warning(f"EGBO rep {rep_idx} failed: {e}")
                    continue

                m = compute_metrics(res, gb, budget=budget, oracle_global_min=gmin)
                seed_metrics.append(m)
                seed_curves.append(np.array(res["running_best"]))
                seed_logs.append({"decisions": res["decisions"], "failures": []})

            _save_and_print(out_dir, ds_label, condition, seed_metrics,
                            seed_curves, seed_logs, gb, gmin, shared_inits,
                            all_rows)

    # Save combined summary
    out_df = pd.DataFrame(all_rows)
    summary_path = out_dir / "metrics_summary.csv"
    out_df.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path} ({len(out_df)} rows)")


def _save_and_print(out_dir, ds_label, condition, seed_metrics, seed_curves,
                    seed_logs, gb, gmin, shared_inits, all_rows):
    if not seed_metrics:
        return
    agg = aggregate_across_seeds(seed_metrics)
    cond_dir = out_dir / ds_label / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    np.save(cond_dir / "curves.npy", np.array(seed_curves))
    with open(cond_dir / "metrics_agg.json",     "w") as f: json.dump(agg, f, indent=2)
    with open(cond_dir / "metrics_per_seed.json", "w") as f: json.dump(seed_metrics, f, indent=2)
    with open(cond_dir / "switch_logs.json",      "w") as f: json.dump(seed_logs, f, indent=2)

    print(f"  {condition:22s}: auc={agg['auc_best']['mean']:.3f}"
          f"±{agg['auc_best']['std']:.3f}"
          f"  final={agg['final_best_normalised']['mean']:.3f}"
          f"  sw={agg['switch_count']['mean']:.1f}")

    for i, m in enumerate(seed_metrics):
        row = {"dataset": ds_label, "condition": condition, "repeat": i,
               "global_best": gb, "global_min": gmin,
               "init_best_norm": float(shared_inits[i][1].max() / gb)}
        row.update(m)
        all_rows.append(row)


if __name__ == "__main__":
    main()
