"""
test_af_generalization.py — regression test for the objective-name
parameterization: the new trust_only/ehvi_approx (looping over
context["objective_names"]) must produce IDENTICAL scores to what the old
hardcoded Tm/kD/viscosity versions would have, when run with the default
objective_names on real excipient data. If this doesn't hold, the
parameterization introduced a behavioural change and should NOT be trusted
on coatings until fixed.

Reuses an existing logged training step (pool_x_norm/pool_pred_mu/
pool_pred_sigma/front_allmax already captured by generate_training_set.py)
rather than fitting a fresh GP — this only needs to check that the AF
CODE's parameterization is correct, not re-derive a new pool.

Usage:
    python test_af_generalization.py
"""

import json
import pathlib
import sys

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
from sandbox import run_af_in_sandbox
from af_interface import SEED_PROGRAMS
from fitness_common import to_allmax

HERE = pathlib.Path(__file__).parent


def load_one_step():
    train_dir = HERE / "training_logs" / "train"
    for f in sorted(train_dir.glob("*.json")):
        log = json.load(open(f))
        for step in log["decisions"]:
            if "pool_x_norm" in step and "pool_pred_sigma" in step:
                Y_running = np.array(log["Y_init"])
                for prior_step in log["decisions"]:
                    if prior_step is step:
                        break
                    if "picked_y" in prior_step:
                        Y_running = np.vstack([Y_running, np.array(prior_step["picked_y"])])
                return step, Y_running
    raise RuntimeError("No usable logged step found in training_logs/train — "
                        "run generate_training_set.py first.")


def main():
    step, Y_running = load_one_step()
    pool_x = np.array(step["pool_x_norm"])
    pool_mu = np.array(step["pool_pred_mu"])
    pool_sigma = np.array(step["pool_pred_sigma"])
    front_allmax = to_allmax(Y_running.copy())
    ref_point_allmax = front_allmax.min(axis=0) - 0.1 * (
        front_allmax.max(axis=0) - front_allmax.min(axis=0) + 1e-9)
    X_obs = np.array([[0.0] * pool_x.shape[1]])  # placeholder, unused by trust_only

    print("Testing parameterized trust_only against direct computation...")
    scores = run_af_in_sandbox(
        SEED_PROGRAMS["trust_only"], pool_x, pool_mu, pool_sigma, X_obs,
        front_allmax, ref_point_allmax, step["step"], 40, step["step"], 0,
        front_allmax,
        objective_names=["Tm", "kD", "viscosity"],
    )
    expected = pool_mu.sum(axis=1)  # what the OLD hardcoded trust_only computed
    max_abs_diff = float(np.max(np.abs(scores - expected)))
    print(f"  max |parameterized - expected| = {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-8, "REGRESSION: parameterized trust_only diverges from expected!"
    print("  PASS: parameterized trust_only matches direct sum(pool_mu, axis=1) exactly.")

    print("\nTesting parameterized ehvi_approx against direct computation...")
    scores2 = run_af_in_sandbox(
        SEED_PROGRAMS["ehvi_approx"], pool_x, pool_mu, pool_sigma, X_obs,
        front_allmax, ref_point_allmax, step["step"], 40, step["step"], 0,
        front_allmax,
        objective_names=["Tm", "kD", "viscosity"],
    )
    # Direct re-derivation of ehvi_approx's logic, objective-name-agnostic
    # by construction here too (this IS the reference implementation, not
    # a copy of the parameterized code under test).
    expected2 = []
    for y in pool_mu:
        dominated = bool(np.any(np.all(front_allmax >= y, axis=1)
                                 & np.any(front_allmax > y, axis=1)))
        vol = float(np.prod(np.maximum(y - ref_point_allmax, 0.0)))
        expected2.append(vol if not dominated else 0.1 * vol)
    expected2 = np.array(expected2)
    max_abs_diff2 = float(np.max(np.abs(scores2 - expected2)))
    print(f"  max |parameterized - expected| = {max_abs_diff2:.2e}")
    assert max_abs_diff2 < 1e-8, "REGRESSION: parameterized ehvi_approx diverges from expected!"
    print("  PASS: parameterized ehvi_approx matches direct re-derivation exactly.")

    print("\nBoth AFs verified equivalent under the objective-name parameterization. "
          "Safe to run on a different oracle (e.g. ada_coatings_oracle).")


if __name__ == "__main__":
    main()
