"""
diagnose_da_coreg_posterior.py — Gate 2 follow-up: is DA-COREG's posterior
itself unreliable on mAb, independent of any acquisition function choice?

Motivation (see §22 of llm_evolved_afs_comprehensive_log.md): ParEGO-EI +
DA-COREG lost to independent-GP baselines on mAb across all 3 replicates
(diff -6.7%/-4.5%/-5.8%, p<0.05 all three, 0/3 directionally positive), and
the Gate 3 evolution run — given the raw obj_correlation signal to work
with — converged on an AF that never uses it. Both results are consistent
with "the DA-COREG surrogate's posterior is just worse on mAb" (independent
of anything downstream built on top of it). This script tests that directly:
fit DA-COREG and independent ModelListGP on the SAME train split, score both
on a held-out split via per-objective predictive NLL and z-score coverage
(calibration), repeated across mAb's aggregation_tendency levels and several
train/test splits.

If DA-COREG's held-out NLL is worse (and/or its calibration is off — e.g.
z-scores not ~N(0,1), overconfident intervals) on mAb specifically, that's a
direct, acquisition-independent explanation for Gate 2/3's negative results,
and further AF-side work on this direction is not well motivated. If NLL/
calibration are comparable or better for DA-COREG, the fault lies specifically
in how obj_correlation is used (or not used) downstream, not the surrogate.

Usage:
    python diagnose_da_coreg_posterior.py --n_splits 20 --n_train 30 --n_test 20
"""

import argparse
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
):
    sys.path.insert(0, str(_p))

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from compose_strategies import _fit_model  # same fit path used by strategy_ablation_cell

AGG_LEVELS = [0.25, 0.50, 0.75, 0.85]


def _to_allmax(Y, directions):
    Y = Y.copy()
    for j, d in enumerate(directions):
        if d == "min":
            Y[:, j] = -Y[:, j]
    return Y


def _nll_and_z(mean, var, y_true):
    """Per-point Gaussian NLL and z-score, both (n, M)."""
    var = np.clip(var, 1e-12, None)
    nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - mean) ** 2 / var
    z = (y_true - mean) / np.sqrt(var)
    return nll, z


def run_one_split(disc, names, directions, n_train, n_test, rng, tkwargs):
    n = len(disc)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_test]

    X_raw = disc._X_raw
    Y_raw = disc._Y_raw
    lo, hi = disc.bounds()[:, 0], disc.bounds()[:, 1]

    X_train_n = (X_raw[train_idx] - lo) / (hi - lo + 1e-12)
    X_test_n = (X_raw[test_idx] - lo) / (hi - lo + 1e-12)
    Y_train = _to_allmax(Y_raw[train_idx], directions)
    Y_test = _to_allmax(Y_raw[test_idx], directions)

    train_x = torch.tensor(X_train_n, **tkwargs)
    train_y = torch.tensor(Y_train, **tkwargs)
    test_x = torch.tensor(X_test_n, **tkwargs)

    results = {}
    for tag, use_da_coreg in (("independent_gp", False), ("da_coreg", True)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = _fit_model(train_x, train_y, use_da_coreg, tkwargs)
                with torch.no_grad():
                    post = model.posterior(test_x)
                    mean = post.mean.detach().cpu().numpy()
                    var = post.variance.clamp_min(1e-12).detach().cpu().numpy()
            except Exception as e:
                results[tag] = {"error": str(e)}
                continue
        nll, z = _nll_and_z(mean, var, Y_test)
        results[tag] = {"nll": nll, "z": z}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_splits", type=int, default=20, help="random train/test splits per agg level")
    ap.add_argument("--n_train", type=int, default=30)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--n_pool", type=int, default=500)
    args = ap.parse_args()

    tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    rng = np.random.default_rng(42)

    all_nll = {"independent_gp": [], "da_coreg": []}
    all_z = {"independent_gp": [], "da_coreg": []}
    n_errors = {"independent_gp": 0, "da_coreg": 0}

    for agg in AGG_LEVELS:
        oracle = MultiObjectiveExcipientOracle(
            protein="mAb_aggregation", tm_noise=0.013, kd_noise=0.096,
            viscosity_noise=0.10, seed=42,
        )
        oracle.protein.aggregation_tendency = agg
        names = oracle.objective_names() if hasattr(oracle, "objective_names") else None
        directions = None

        print(f"agg={agg}...", end="", flush=True)
        for split_idx in range(args.n_splits):
            disc = oracle.make_discrete_oracle(n_samples=args.n_pool, seed=int(rng.integers(1e6)))
            if directions is None:
                directions = disc.objective_directions()
                names = disc.objective_names()
            res = run_one_split(disc, names, directions, args.n_train, args.n_test, rng, tkwargs)
            for tag in ("independent_gp", "da_coreg"):
                if "error" in res[tag]:
                    n_errors[tag] += 1
                    continue
                all_nll[tag].append(res[tag]["nll"])
                all_z[tag].append(res[tag]["z"])
            print(".", end="", flush=True)
        print()

    print("\n" + "=" * 70)
    print("DA-COREG vs independent-GP held-out posterior diagnostic (mAb)")
    print("=" * 70)
    for tag in ("independent_gp", "da_coreg"):
        if not all_nll[tag]:
            print(f"{tag}: all fits errored ({n_errors[tag]})")
            continue
        nll_stack = np.concatenate(all_nll[tag], axis=0)  # (n_points, M)
        z_stack = np.concatenate(all_z[tag], axis=0)
        mean_nll_per_obj = nll_stack.mean(axis=0)
        z_std_per_obj = z_stack.std(axis=0)
        z_cov68 = (np.abs(z_stack) <= 1.0).mean(axis=0)  # ideal ~0.68 if calibrated
        print(f"\n{tag}  (n_fits_errored={n_errors[tag]}, n_points={nll_stack.shape[0]})")
        for j, name in enumerate(names):
            print(f"  {name:12s}  mean NLL={mean_nll_per_obj[j]:+.4f}  "
                  f"z_std={z_std_per_obj[j]:.3f} (ideal 1.0)  "
                  f"|z|<=1 coverage={z_cov68[j]:.3f} (ideal ~0.68)")
        print(f"  ALL OBJS      mean NLL={mean_nll_per_obj.mean():+.4f}")

    if all_nll["independent_gp"] and all_nll["da_coreg"]:
        ig = np.concatenate(all_nll["independent_gp"], axis=0).mean()
        dc = np.concatenate(all_nll["da_coreg"], axis=0).mean()
        print(f"\nOverall mean NLL: independent_gp={ig:+.4f}  da_coreg={dc:+.4f}  "
              f"(lower is better; da_coreg {'WORSE' if dc > ig else 'better'} by {abs(dc - ig):.4f})")
        print("\nInterpretation: if da_coreg's NLL is clearly worse and/or its z_std is far "
              "from 1.0 (overconfident if <1, underconfident if >1) specifically on mAb, "
              "that's an acquisition-independent explanation for Gate 2/3's negative "
              "results — the surrogate itself is the problem here, not how obj_correlation "
              "is used downstream.")


if __name__ == "__main__":
    main()
