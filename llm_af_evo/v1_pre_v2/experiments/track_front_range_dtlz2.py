"""
track_front_range_dtlz2.py — does pareto_front_range shrink (the
self-annealing premise front-range normalisation needs) or grow (as §22
found on the tunable synthetic domain, and as track_front_range_coatings.py
checked on the real coatings domain) over a campaign on DTLZ2 — the
canonical synthetic benchmark, distinct from both the tunable-plateau
domain and the two real domains (mAb, coatings) already checked?

Deliberately duplicates track_front_range_coatings.py's run_one_tracked
loop verbatim (same _queried reset, same _front_range convention) with
only the oracle swapped for DiscreteSyntheticMOOracle.build_dtlz2() — a
read-only diagnostic, not a change to shared campaign semantics.

Usage:
    python track_front_range_dtlz2.py --n_campaigns 5
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
):
    sys.path.insert(0, str(_p))
from excipient_campaign_mo import pareto_front_of, snap_query_mo, make_shared_inits

import torch
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from full_replay import strategy_evolved_af

HERE = pathlib.Path(__file__).parent

# Same two AF code strings as track_front_range.py / track_front_range_coatings.py,
# kept in sync by hand.
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
    ap.add_argument("--n_obj", type=int, default=3, help="DTLZ2 objective count.")
    ap.add_argument("--out_path", default=str(HERE / "front_range_trace_dtlz2.json"))
    args = ap.parse_args()

    oracle = DiscreteSyntheticMOOracle.build_dtlz2(n_obj=args.n_obj)
    print(f"DTLZ2 oracle: {len(oracle)} pool samples, "
          f"objectives={oracle.objective_names()}, budget={args.budget}, "
          f"n_campaigns={args.n_campaigns}")

    inits = make_shared_inits(oracle, args.n_campaigns, args.n_init, rng_seed=args.seed)

    results = {}
    for cond_name, cfg in CONDITIONS.items():
        traces = []
        for camp_i, (X_init, Y_init) in enumerate(inits):
            trace = run_one_tracked(oracle, X_init, Y_init, args.budget,
                                     args.batch_size, seed=camp_i,
                                     af_code=cfg["af_code"])
            traces.append(trace)
        results[cond_name] = traces

    names = oracle.objective_names()
    print(f"\nMean front_range per batch (init-only -> final), by condition:")
    for cond_name, traces in results.items():
        n_batches = len(traces[0])
        print(f"\n  {cond_name}:")
        for name in names:
            vals_by_batch = [
                np.mean([t[b]["front_range"][name] for t in traces])
                for b in range(n_batches)
            ]
            first, last = vals_by_batch[0], vals_by_batch[-1]
            pct = 100.0 * (last - first) / max(first, 1e-9)
            print(f"    {name:>16}: {first:.4g} -> {last:.4g} ({pct:+.1f}%)")

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full per-batch trace to {args.out_path}")


if __name__ == "__main__":
    main()
