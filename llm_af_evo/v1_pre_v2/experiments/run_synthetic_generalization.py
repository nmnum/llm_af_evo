"""
run_synthetic_generalization.py — cross-domain generalization test for a
final_population.json's evolved AFs (e.g. evolution_runs/run_v2_coatings_
real_100/final_population.json) against synthetic multi-objective
benchmarks with known regime characteristics, on the live
DiscreteSyntheticMOOracle (ZDT1 / DTLZ2) — no held-out-file split needed,
unlike the coatings/mAb tracks, since a synthetic oracle can be freely
re-instantiated with fresh random draws per campaign rather than requiring
a fixed real dataset to hold data out of.

Modeled directly on run_coatings_generalization.py's pattern (same env-var
BLAS-determinism fix, same make_shared_inits + run_mo_campaign +
strategy_evolved_af usage, same small-pilot-first caution) — see that
file's docstring for why the env vars must be set before numpy/scipy/torch
import.

Two domains available, chosen to bracket the regimes the Aqeeli et al.
"Novelty-aware evolutionary Bayesian optimisation" paper found most
different in its own synthetic benchmarks (see research context from
this project's design discussions):
  zdt1  — 2-objective, smooth, convex front. The "boring" contrast case
          where that paper found hybrid/novelty methods barely differ
          from acquisition-only. If our AFs also show little separation
          here, that's consistent, not a failure of the test.
  dtlz2 — many-objective (n_obj configurable, default 5) — the regime
          where that paper found the LARGEST gains for exploration/
          diversity-promoting methods. The sharpest test of whether
          front-loaded uncertainty (the design pattern found on coatings,
          only partially replicated on mAb) matters more here.

Usage:
    python run_synthetic_generalization.py --domain zdt1 --n_campaigns 3    # pilot
    python run_synthetic_generalization.py --domain zdt1 --n_campaigns 20   # real run
    python run_synthetic_generalization.py --domain dtlz2 --n_obj 5 --n_campaigns 20
    # --population_path defaults to the coatings n_seeds=100 run's final population
"""

import os

# MUST be set before numpy/scipy/torch are imported anywhere in the
# process — see run_coatings_generalization.py's identical block for the
# directly-verified rationale (torch.set_num_threads(1) alone is
# insufficient; scipy's L-BFGS-B / numpy's BLAS backend need the env vars
# set before their own C-extension init).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from full_replay import strategy_evolved_af

HERE = pathlib.Path(__file__).parent


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)  # same reproducibility fix as full_replay.py
    kwargs = dict(kwargs)
    if fn is strategy_evolved_af:
        kwargs["budget"] = budget
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["zdt1", "dtlz2"], default="zdt1")
    ap.add_argument("--n_obj", type=int, default=5,
                     help="DTLZ2 only — number of objectives (default 5, "
                          "matching the many-objective regime where prior "
                          "literature found the largest hybrid/novelty gains).")
    ap.add_argument("--n_campaigns", type=int, default=3,
                     help="Default is a small pilot — check timing before scaling to 20.")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--population_path",
                     default=str(HERE / "evolution_runs" / "run_v2_coatings_real_100" /
                                 "final_population.json"))
    ap.add_argument("--out_path", default=None,
                     help="Defaults to synthetic_generalization_<domain>_results.json")
    args = ap.parse_args()

    out_path = args.out_path or str(
        HERE / f"synthetic_generalization_{args.domain}_results.json")

    if args.domain == "zdt1":
        oracle = DiscreteSyntheticMOOracle.build_zdt1()
    else:
        oracle = DiscreteSyntheticMOOracle.build_dtlz2(n_obj=args.n_obj)
    print(f"Oracle: {args.domain} ({args.n_obj if args.domain == 'dtlz2' else 2} obj), "
          f"{len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")

    population = json.load(open(args.population_path))
    print(f"Loaded {len(population)} evolved AFs from {args.population_path}")

    conditions = {"mo_egbo_novelty": (strategy_mo_egbo_novelty, {})}
    for p in population:
        conditions[p["id"]] = (strategy_evolved_af, {"af_code": p["code"]})

    inits = make_shared_inits(oracle, args.n_campaigns, args.n_init, rng_seed=args.seed)

    results = {cond: [] for cond in conditions}
    for cond_name, (fn, kwargs) in conditions.items():
        print(f"\n{cond_name}...")
        for i, (X_init, Y_init) in enumerate(inits):
            t0 = time.perf_counter()
            final_hv = run_one(oracle, X_init, Y_init, args.budget, args.batch_size,
                                i, fn, kwargs)
            elapsed = time.perf_counter() - t0
            n_batches = max(1, (args.budget - args.n_init) // args.batch_size)
            results[cond_name].append(final_hv)
            print(f"  campaign {i}: final_hv={final_hv:.4g}  "
                  f"{elapsed:.1f}s total ({elapsed/n_batches:.2f}s/batch)")

    baseline = np.array(results["mo_egbo_novelty"])
    print(f"\n{'='*60}")
    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small (max achievable two-sided p at n={args.n_campaigns} is "
              f"{2*(0.5**args.n_campaigns):.4f}). The numbers below are for a "
              f"timing/smoke check only — do not interpret them, rerun with "
              f"--n_campaigns 20 for anything conclusive.\n")
    print(f"Baseline (mo_egbo_novelty) mean final HV: {baseline.mean():.4g}")
    ranked = []
    for cond_name in [c for c in conditions if c != "mo_egbo_novelty"]:
        af_hv = np.array(results[cond_name])
        diffs = af_hv - baseline
        pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        ranked.append((cond_name, af_hv.mean(), pct, n_wins, pval))
        print(f"{cond_name:<16} mean_hv={af_hv.mean():.4g}  diff={pct:+.1f}%  "
              f"wins={n_wins}/{args.n_campaigns}  p={pval:.4f}")

    ranked.sort(key=lambda r: -r[1])
    print(f"\nRanked by mean HV:")
    for cond_name, mean_hv, pct, n_wins, pval in ranked:
        print(f"  {cond_name:<16} mean_hv={mean_hv:.4g}  diff={pct:+.1f}%  p={pval:.4f}")

    if args.n_campaigns < 20:
        print(f"\nThis was a {args.n_campaigns}-campaign PILOT — check the per-batch "
              f"timing above, then rerun with --n_campaigns 20 for the real comparison.")

    with open(out_path, "w") as f:
        json.dump({
            "domain": args.domain, "n_obj": args.n_obj if args.domain == "dtlz2" else 2,
            "n_campaigns": args.n_campaigns, "budget": args.budget,
            "n_init": args.n_init, "batch_size": args.batch_size, "seed": args.seed,
            "objective_names": oracle.objective_names(),
            "objective_directions": oracle.objective_directions(),
            "population_path": args.population_path,
            "per_campaign_final_hv": results,
        }, f, indent=2)
    print(f"\nSaved per-campaign results to {out_path}")


if __name__ == "__main__":
    main()
