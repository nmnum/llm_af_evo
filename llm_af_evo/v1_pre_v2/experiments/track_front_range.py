"""
track_front_range.py — instrumented replay of a few campaigns on the
tunable synthetic domain, logging pareto_front_range PER BATCH (not just
final HV) for gen6_child0_tuned vs hint_fixed_ucb.

Purpose: the paired-Wilcoxon comparison of the two conditions (run via
run_tunable_domain_generalization.py) came back non-significant (12/20,
p=0.15) despite both individually beating trust_only. The suspected reason:
front-range normalisation's real differentiator vs a fixed-beta raw-sigma
UCB is that it's IMPLICITLY self-annealing — sigma/front_range grows in
relative weight as the front shrinks, without an explicit progress-decay
term. If front_range barely shrinks over a campaign on this domain, the two
AFs are nearly the same function (sigma_norm ~= sigma * const), which would
exactly explain a wash. This script checks that directly instead of
guessing.

Deliberately duplicates run_mo_campaign's loop (same _queried reset, same
fixed HV reference point, same pareto_front_of/directions convention) rather
than modifying the shared harness — this is a read-only diagnostic on a
copy of the loop, not a change to campaign semantics anyone else depends on.

Usage:
    python track_front_range.py --n_campaigns 5
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
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
from excipient_campaign_mo import pareto_front_of, snap_query_mo, make_shared_inits

import torch
from tunable_synthetic_oracle import TunableSyntheticMOOracle
from full_replay import strategy_evolved_af

HERE = pathlib.Path(__file__).parent

# Same two AF code strings as run_tunable_domain_generalization.py (kept in
# sync by hand — this is a throwaway diagnostic script, not a shared module).
SEED_HINT_FIXED_UCB = '''
def score_pool(context):
    """Fixed-weight UCB, RAW (non-front-range-normalised) sigma."""
    names = context["objective_names"]
    beta = 2.0
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + beta * sigma_sum)
    return scores
'''.strip("\n")

_GEN6_CHILD0_TUNED = '''
def score_pool(context):
    """Front-range-normalised sigma UCB (gen6_child0). beta=15.0."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    beta = 15.0
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''.strip("\n")

CONDITIONS = {
    "hint_fixed_ucb": {"af_code": SEED_HINT_FIXED_UCB},
    "gen6_child0_tuned": {"af_code": _GEN6_CHILD0_TUNED},
}


def _front_range(Y_obs: np.ndarray, directions) -> np.ndarray:
    """Per-objective range (max-min) of the current non-dominated set —
    same convention sweep_tunable_domain.py's _pareto_front_range uses and
    the same one score_pool's context["pareto_front_range"] is built from."""
    pf_idx = pareto_front_of(Y_obs, directions=directions)
    front = Y_obs[pf_idx]
    if len(front) < 2:
        return np.ptp(Y_obs, axis=0)
    return np.ptp(front, axis=0)


def run_one_tracked(oracle, X_init, Y_init, budget, batch_size, seed, af_code):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bounds = oracle.bounds()
    directions = oracle.objective_directions()
    names = oracle.objective_names()

    X_obs, Y_obs = X_init.copy(), Y_init.copy()

    oracle._queried = set()
    X_all_s = oracle._scaler.transform(oracle._X_raw)
    X_init_s = oracle._scaler.transform(X_init)
    for row in X_init_s:
        idx = int(np.argmin(np.linalg.norm(X_all_s - row, axis=1)))
        oracle._queried.add(idx)

    n_batches = max(1, (budget - len(X_init)) // batch_size)

    # front_range BEFORE any batches run (init-only front)
    fr0 = _front_range(Y_obs, directions)
    trace = [{"batch": 0, "n_obs": len(Y_obs),
              "front_range": {n: float(v) for n, v in zip(names, fr0)}}]

    for b in range(n_batches):
        candidates, _ = strategy_evolved_af(
            oracle=oracle, X_obs=X_obs, Y_obs=Y_obs, bounds=bounds,
            batch_size=batch_size, rng=rng, af_code=af_code, budget=budget)
        new_x, new_y = snap_query_mo(candidates[:batch_size], oracle)
        if new_x:
            X_obs = np.vstack([X_obs, new_x])
            Y_obs = np.vstack([Y_obs, new_y])
        fr = _front_range(Y_obs, directions)
        trace.append({"batch": b + 1, "n_obs": len(Y_obs),
                       "front_range": {n: float(v) for n, v in zip(names, fr)}})

    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_campaigns", type=int, default=5)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plateau_sharpness", type=float, default=5.0)
    ap.add_argument("--noise_level", type=float, default=0.08)
    ap.add_argument("--noise_mode", default="proportional")
    ap.add_argument("--scale2", type=float, default=3.0)
    ap.add_argument("--out_path", default=str(HERE / "front_range_trace.json"))
    args = ap.parse_args()

    oracle = TunableSyntheticMOOracle.build(
        plateau_sharpness=args.plateau_sharpness, noise_level=args.noise_level,
        noise_mode=args.noise_mode, scale2=args.scale2, seed=args.seed)
    names = oracle.objective_names()

    inits = make_shared_inits(oracle, args.n_campaigns, args.n_init, rng_seed=args.seed)

    all_traces = {cond: [] for cond in CONDITIONS}
    for cond_name, kwargs in CONDITIONS.items():
        print(f"\n{cond_name}...")
        for i, (X_init, Y_init) in enumerate(inits):
            trace = run_one_tracked(oracle, X_init, Y_init, args.budget,
                                     args.batch_size, i, kwargs["af_code"])
            all_traces[cond_name].append(trace)
            fr_start = trace[0]["front_range"]
            fr_end = trace[-1]["front_range"]
            shrink = {n: (1 - fr_end[n] / fr_start[n]) * 100 if fr_start[n] > 0 else float("nan")
                      for n in names}
            print(f"  campaign {i}: front_range[0]={fr_start}  "
                  f"front_range[-1]={fr_end}  shrink%={ {k: f'{v:.1f}' for k,v in shrink.items()} }")

    # Aggregate mean front_range per batch index, per objective, per condition
    print(f"\n{'='*70}\nMean front_range by batch index (averaged over {args.n_campaigns} campaigns):")
    for cond_name, traces in all_traces.items():
        n_batches = len(traces[0])
        print(f"\n{cond_name}:")
        print(f"  {'batch':>6}", *[f"{n:>12}" for n in names])
        for b in range(n_batches):
            means = {n: np.mean([t[b]["front_range"][n] for t in traces]) for n in names}
            print(f"  {b:>6}", *[f"{means[n]:>12.4f}" for n in names])

    with open(args.out_path, "w") as f:
        json.dump({"n_campaigns": args.n_campaigns, "budget": args.budget,
                    "n_init": args.n_init, "batch_size": args.batch_size,
                    "objective_names": names, "traces": all_traces}, f, indent=2)
    print(f"\nSaved per-batch front_range traces to {args.out_path}")


if __name__ == "__main__":
    main()
