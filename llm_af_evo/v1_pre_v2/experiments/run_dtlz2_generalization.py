"""
run_dtlz2_generalization.py — the confirmation experiment §25's front-range
diagnostic set up but didn't answer: does front-range-normalised sigma-
weighting (gen6_child0) beat a well-tuned baseline (trust_only) and a
random-scoring floor, on DTLZ2 — the canonical synthetic MO benchmark,
distinct from both the tunable-plateau domain (§22) and the two real
domains (mAb, coatings) already checked?

§25 (track_front_range_dtlz2.py) confirmed front_range GROWS on DTLZ2 too
(+27-44%, matching the tunable-domain and coatings results), which predicts
the mechanism should fail here just as it failed on those two — but growth
alone is a proxy, not the actual question. This script runs the real
closed-loop HV comparison to check the mechanism's win/loss directly,
rather than inferring it from the front_range trend.

Five conditions, same as run_tunable_domain_generalization.py:
  random               — pure random score. Sanity floor.
  trust_only            — pure exploitation (mu_sum only). The baseline.
  hint_fixed_ucb        — mu_sum + 2.0*sigma_sum, RAW (non-normalised)
                           sigma. The literature-standard UCB this needs to
                           beat to say front-range normalisation specifically
                           (not just "any uncertainty bonus") is doing the work.
  gen6_child0_beta0.5    — the AS-EVOLVED coefficient (coatings n=75 winner),
                           unchanged, so the comparison isn't quietly
                           re-tuning the AF to fit the new domain.
  gen6_child0_tuned      — beta set to 15.0, matching the coefficient used
                           in track_front_range_dtlz2.py's diagnostic, for
                           direct comparability with that trace.

Adapted directly from run_tunable_domain_generalization.py — same
run_mo_campaign harness, same Wilcoxon signed-rank comparison against
trust_only, same n<20-is-a-pilot-only warning. Swaps
TunableSyntheticMOOracle for DiscreteSyntheticMOOracle.build_dtlz2().

Usage:
    python run_dtlz2_generalization.py --n_campaigns 3    # pilot
    python run_dtlz2_generalization.py --n_campaigns 20   # real comparison
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
import time

import numpy as np
from scipy.stats import wilcoxon

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
from excipient_campaign_mo import run_mo_campaign, make_shared_inits

import torch
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from full_replay import strategy_evolved_af
from af_interface import SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent

SEED_RANDOM = '''
def score_pool(context):
    """Pure random score — sanity floor. If nothing beats this, nothing
    else in the comparison is meaningful."""
    rng = np.random.default_rng(0)
    return list(rng.random(len(context["pool"])))
'''.strip("\n")

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

_GEN6_CHILD0_TEMPLATE = '''
def score_pool(context):
    """Front-range-normalised sigma UCB (gen6_child0). beta={beta}."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    beta = {beta}
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''.strip("\n")

_GEN6_CHILD0_FROZEN_DENOM = '''
def score_pool(context):
    """Front-range-normalised sigma UCB, FROZEN denominator (§25 follow-up:
    divides by the front's range AT INIT ONLY, context["pareto_front_range_init"],
    not the growing per-batch context["pareto_front_range"] gen6_child0 uses
    — directly targets §23's root cause (the denominator growing over the
    campaign is what breaks the self-annealing premise) while keeping the
    cross-objective scale normalisation. beta={beta}."""
    names = context["objective_names"]
    front_range_init = context["pareto_front_range_init"]
    beta = {beta}
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range_init[name] for name in names)
        scores.append(mu_sum + beta * sigma_norm)
    return scores
'''.strip("\n")

CONDITIONS = {
    "random": (strategy_evolved_af, {"af_code": SEED_RANDOM}),
    "trust_only": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["trust_only"]}),
    "hint_fixed_ucb": (strategy_evolved_af, {"af_code": SEED_HINT_FIXED_UCB}),
    "gen6_child0_beta0.5": (strategy_evolved_af,
                             {"af_code": _GEN6_CHILD0_TEMPLATE.format(beta=0.5)}),
    "gen6_child0_tuned": (strategy_evolved_af,
                           {"af_code": _GEN6_CHILD0_TEMPLATE.format(beta=15.0)}),
    "gen6_child0_frozen_denom": (strategy_evolved_af,
                                  {"af_code": _GEN6_CHILD0_FROZEN_DENOM.format(beta=15.0)}),
}
BASELINE = "trust_only"


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs, n_init=None):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn is strategy_evolved_af:
        kwargs["budget"] = budget
        kwargs["n_init"] = n_init  # only gen6_child0_frozen_denom reads this;
                                    # harmless no-op for every other condition
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_campaigns", type=int, default=3,
                     help="Default is a small pilot — check timing before scaling to 20.")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_obj", type=int, default=3, help="DTLZ2 objective count.")
    ap.add_argument("--out_path", default=str(HERE / "dtlz2_generalization_results.json"))
    args = ap.parse_args()

    oracle = DiscreteSyntheticMOOracle.build_dtlz2(n_obj=args.n_obj)
    print(f"Oracle: {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")

    inits = make_shared_inits(oracle, args.n_campaigns, args.n_init, rng_seed=args.seed)

    results = {cond: [] for cond in CONDITIONS}
    for cond_name, (fn, kwargs) in CONDITIONS.items():
        print(f"\n{cond_name}...")
        for i, (X_init, Y_init) in enumerate(inits):
            t0 = time.perf_counter()
            final_hv = run_one(oracle, X_init, Y_init, args.budget, args.batch_size,
                                i, fn, kwargs, n_init=args.n_init)
            elapsed = time.perf_counter() - t0
            n_batches = max(1, (args.budget - args.n_init) // args.batch_size)
            results[cond_name].append(final_hv)
            print(f"  campaign {i}: final_hv={final_hv:.4g}  "
                  f"{elapsed:.1f}s total ({elapsed/n_batches:.2f}s/batch)")

    baseline = np.array(results[BASELINE])
    print(f"\n{'='*60}")
    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small (max achievable two-sided p at n=3 is 0.25). The numbers below "
              f"are for a timing/smoke check only — rerun with --n_campaigns 20 for "
              f"anything conclusive.\n")
    print(f"Baseline ({BASELINE}) mean final HV: {baseline.mean():.4g}")
    for cond_name in [c for c in CONDITIONS if c != BASELINE]:
        af_hv = np.array(results[cond_name])
        diffs = af_hv - baseline
        pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        print(f"{cond_name:<20} mean_hv={af_hv.mean():.4g}  diff={pct:+.1f}%  "
              f"wins={n_wins}/{args.n_campaigns}  p={pval:.4f}")

    if args.n_campaigns < 20:
        print(f"\nThis was a {args.n_campaigns}-campaign PILOT — check the per-batch "
              f"timing above, then rerun with --n_campaigns 20 for the real comparison.")

    with open(args.out_path, "w") as f:
        json.dump({
            "n_campaigns": args.n_campaigns, "budget": args.budget,
            "n_init": args.n_init, "batch_size": args.batch_size, "seed": args.seed,
            "n_obj": args.n_obj,
            "objective_names": oracle.objective_names(),
            "objective_directions": oracle.objective_directions(),
            "baseline_condition": BASELINE,
            "per_campaign_final_hv": results,
        }, f, indent=2)
    print(f"\nSaved per-campaign results to {args.out_path}")


if __name__ == "__main__":
    main()
