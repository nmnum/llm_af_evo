"""
debug_compose.py — one-shot diagnostic for the compose_batch/DA-COREG
infrastructure, same purpose as debug_generation.py: surface REAL
exceptions/tracebacks directly, since strategy_ablation_cell's broad
except-and-fallback would otherwise swallow them the same way
strategy_evolved_generation did before the numpy.random sandbox bug was
found. Run this BEFORE run_synthetic_compose_pilot.py.

Usage:
    python debug_compose.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pathlib
import sys
import traceback

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

from excipient_campaign_mo import make_shared_inits, pareto_front_of
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from compose_sandbox import run_compose_in_sandbox, ComposeSandboxError
from compose_interface import COMPOSE_SEED_PROGRAMS
from fitness_common import to_allmax

oracle = DiscreteSyntheticMOOracle.build_zdt1()
bounds = oracle.bounds()
d = bounds.shape[0]
lo, hi = bounds[:, 0], bounds[:, 1]

inits = make_shared_inits(oracle, 1, 10, rng_seed=42)
X_init, Y_init = inits[0]
X_norm = (X_init - lo) / (hi - lo + 1e-12)

directions = oracle.objective_directions()
pf_idx_np = pareto_front_of(Y_init, directions=directions)
front_allmax = to_allmax(Y_init.copy(), directions=directions)

# ── Part 1: compose_sandbox / greedy_marginal_hv seed, in isolation ────────
print("=" * 60)
print("Part 1: run_compose_in_sandbox with greedy_marginal_hv")
print("=" * 60)
n_cand = 25
rng = np.random.default_rng(0)
pool_x = rng.random((n_cand, d))
pool_mu = np.column_stack([
    rng.normal(Y_init[:, j].mean(), Y_init[:, j].std() + 1e-6, n_cand)
    if directions[j] == "max" else
    -rng.normal(Y_init[:, j].mean(), Y_init[:, j].std() + 1e-6, n_cand)
    for j in range(Y_init.shape[1])
])
pool_sigma = np.abs(rng.normal(0.1, 0.05, (n_cand, Y_init.shape[1]))) + 1e-6
ref_point_allmax = front_allmax.min(axis=0) - 0.1 * (
    front_allmax.max(axis=0) - front_allmax.min(axis=0) + 1e-9) if len(front_allmax) > 0 \
    else -np.ones(Y_init.shape[1])

for seed_name in ["greedy_marginal_hv", "greedy_marginal_hv_mc"]:
    print(f"\n--- {seed_name} ---")
    try:
        idx = run_compose_in_sandbox(
            COMPOSE_SEED_PROGRAMS[seed_name], pool_x, pool_mu, pool_sigma,
            X_norm, front_allmax, ref_point_allmax, step=10, budget=40, n_obs=10,
            stagnant_batches=0, k=5, objective_names=oracle.objective_names(),
        )
        print(f"OK — selected indices {idx}")
    except ComposeSandboxError as e:
        print(f"ComposeSandboxError: {e}")
    except Exception:
        traceback.print_exc()

# ── Part 2: da_coreg fit + posterior, in isolation ──────────────────────────
print("\n" + "=" * 60)
print("Part 2: fit_da_coreg_model")
print("=" * 60)
try:
    from da_coreg import fit_da_coreg_model
    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    Y_int = Y_init.copy()
    for j, direction in enumerate(directions):
        if direction == "min":
            Y_int[:, j] = -Y_int[:, j]
    train_x = torch.tensor(X_norm, **tkwargs)
    train_y = torch.tensor(Y_int, **tkwargs)
    da_model = fit_da_coreg_model(train_x, train_y, tkwargs)
    candidates = torch.tensor(rng.random((5, d)), **tkwargs)
    post = da_model.posterior(candidates)
    print(f"OK — posterior mean shape {tuple(post.mean.shape)}, "
          f"variance shape {tuple(post.variance.shape)}")
except Exception:
    traceback.print_exc()

# ── Part 3: full strategy_ablation_cell, all 4 combinations ────────────────
print("\n" + "=" * 60)
print("Part 3: strategy_ablation_cell, all 4 (da_coreg x compose) combinations")
print("=" * 60)
from compose_strategies import strategy_ablation_cell

for use_da_coreg in [False, True]:
    for use_compose in [False, True]:
        label = f"da_coreg={use_da_coreg}, compose={use_compose}"
        print(f"\n--- {label} ---")
        kwargs = {"use_da_coreg": use_da_coreg, "use_compose": use_compose, "budget": 40}
        if use_compose:
            kwargs["compose_code"] = COMPOSE_SEED_PROGRAMS["greedy_marginal_hv"]
        try:
            rng2 = np.random.default_rng(1)
            new_x, extra = strategy_ablation_cell(
                oracle, X_init, Y_init, bounds, batch_size=5, rng=rng2, **kwargs)
            print(f"OK — new_x shape {new_x.shape}, extra={extra}")
        except Exception:
            traceback.print_exc()
