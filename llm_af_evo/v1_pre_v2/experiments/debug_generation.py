"""
debug_generation.py — one-shot diagnostic for the generation pilot's
suspicious all-identical, all-negative results (see the run
transcript: trust_only_gen_lhs/perturb/hybrid all produced bit-identical
per-replicate percentages, the signature of strategy_evolved_generation
silently falling back to strategy_mo_egbo on every call). This script
calls run_generator_in_sandbox DIRECTLY (bypassing strategy_evolved_
generation's broad except-and-fallback) so the real exception/traceback
surfaces instead of being swallowed.

Usage:
    python debug_generation.py
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

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import make_shared_inits, pareto_front_of
from gen_sandbox import run_generator_in_sandbox, GenSandboxError
from gen_interface import GEN_SEED_PROGRAMS

PROTEIN = "mAb_aggregation"

oracle_full = MultiObjectiveExcipientOracle(
    protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
oracle = oracle_full.make_discrete_oracle(n_samples=500, seed=42)
bounds = oracle.bounds()
d = bounds.shape[0]

inits = make_shared_inits(oracle, 1, 10, rng_seed=42)
X_init, Y_init = inits[0]
lo, hi = bounds[:, 0], bounds[:, 1]
X_norm = (X_init - lo) / (hi - lo + 1e-12)

directions = oracle.objective_directions()
pf_idx_np = pareto_front_of(Y_init, directions=directions)
pareto_x_n = X_norm[pf_idx_np] if len(pf_idx_np) > 0 else np.zeros((0, d))

print(f"X_norm shape: {X_norm.shape}, pareto_x_n shape: {pareto_x_n.shape}, d={d}")

for name, code in GEN_SEED_PROGRAMS.items():
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    try:
        out = run_generator_in_sandbox(
            code, X_norm, pareto_x_n, step=10, budget=40, n_obs=10,
            stagnant_batches=0, n_dims=d, n_candidates=20,
        )
        print(f"OK — shape {out.shape}, range [{out.min():.3f}, {out.max():.3f}]")
    except GenSandboxError as e:
        print(f"GenSandboxError: {e}")
    except Exception:
        print("Unexpected exception (not GenSandboxError):")
        traceback.print_exc()
