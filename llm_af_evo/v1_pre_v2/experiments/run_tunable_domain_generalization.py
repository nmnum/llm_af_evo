"""
run_tunable_domain_generalization.py — the confirmation experiment the
thesis's diagnosis specifies: does front-range-normalised sigma-weighting
(gen6_child0) beat a well-tuned baseline (trust_only) and a random-scoring
floor, on a domain built to avoid coatings' mu_sum-dominance failure and
mAb's noise-floor failure (see tunable_synthetic_oracle.py + sweep_
tunable_domain.py, which picked plateau_sharpness=5.0/noise_level=0.08/
scale2=3.0 as the working point: dominance_ratio~11-17, achieved_cv~0.06,
front_range_ratio~2.3-3.4).

Four conditions:
  random          — pure random score. Sanity floor: if a "smart" AF
                     doesn't beat this, nothing below it is meaningful.
  trust_only       — pure exploitation (mu_sum only). The "well-tuned
                     baseline" the thesis couldn't yet beat on coatings/mAb.
  hint_fixed_ucb   — mu_sum + 2.0*sigma_sum, RAW (non-normalised) sigma.
                     The literature-standard UCB this experiment needs to
                     beat to say front-range normalisation specifically
                     (not just "any uncertainty bonus") is what's doing
                     the work.
  gen6_child0      — mu_sum + beta*sigma_norm, front-range-normalised
                     sigma. Two beta variants are run because the domain
                     sweep found dominance_ratio~11-17 here (vs coatings'
                     regime where beta=0.5 was the n=75-validated winner):
                       gen6_child0_beta0.5  — the AS-EVOLVED coefficient,
                         unchanged from the coatings run. Included so the
                         comparison isn't quietly re-tuning the AF to fit
                         the new domain.
                       gen6_child0_tuned    — beta set to ~dominance_ratio
                         (15.0), i.e. the coefficient the sweep says is
                         needed for the normalised sigma term to have
                         parity with mu_sum on THIS domain. This is the
                         fair test of the mechanism itself, decoupled from
                         whichever beta the LLM happened to evolve on a
                         differently-scaled domain.

Adapted directly from run_coatings_generalization.py — same run_mo_campaign
harness, same Wilcoxon signed-rank comparison against a baseline, same
n<20-is-a-pilot-only warning. Swaps DiscreteADACoatingsOracle for
TunableSyntheticMOOracle and mo_egbo_novelty for trust_only as the
baseline (this domain has no analogue of the real mo_egbo_novelty
strategy — trust_only IS the "well-tuned baseline" here, matching the
thesis's own framing of what wasn't yet beaten).

Usage:
    python run_tunable_domain_generalization.py --n_campaigns 3    # pilot
    python run_tunable_domain_generalization.py --n_campaigns 20   # real comparison
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
from tunable_synthetic_oracle import TunableSyntheticMOOracle
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

# Generic (objective_names-loop, not hardcoded Tm/kD/viscosity) version of
# the log's hint_fixed_ucb: mu_sum + 2.0*sigma_sum, RAW sigma — the
# literature-standard UCB baseline for "any uncertainty bonus helps".
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

# gen6_child0 as validated on coatings n=75 (docs/llm_evolved_afs_
# comprehensive_log.md section 6): mu_sum + 0.5*sigma_norm.
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

CONDITIONS = {
    "random": (strategy_evolved_af, {"af_code": SEED_RANDOM}),
    "trust_only": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["trust_only"]}),
    "hint_fixed_ucb": (strategy_evolved_af, {"af_code": SEED_HINT_FIXED_UCB}),
    "gen6_child0_beta0.5": (strategy_evolved_af,
                             {"af_code": _GEN6_CHILD0_TEMPLATE.format(beta=0.5)}),
    "gen6_child0_tuned": (strategy_evolved_af,
                           {"af_code": _GEN6_CHILD0_TEMPLATE.format(beta=15.0)}),
}
BASELINE = "trust_only"


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn is strategy_evolved_af:
        kwargs["budget"] = budget
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
    # Domain params — defaults are sweep_tunable_domain.py's chosen working
    # point (dominance_ratio~11-17, achieved_cv~0.06, front_range_ratio~2.3-3.4).
    ap.add_argument("--plateau_sharpness", type=float, default=5.0)
    ap.add_argument("--noise_level", type=float, default=0.08)
    ap.add_argument("--noise_mode", default="proportional",
                     choices=["homoscedastic", "proportional", "input_dependent"])
    ap.add_argument("--scale2", type=float, default=3.0)
    ap.add_argument("--out_path", default=str(HERE / "tunable_domain_generalization_results.json"))
    args = ap.parse_args()

    oracle = TunableSyntheticMOOracle.build(
        plateau_sharpness=args.plateau_sharpness, noise_level=args.noise_level,
        noise_mode=args.noise_mode, scale2=args.scale2, seed=args.seed)
    print(f"Oracle: {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()}), plateau_sharpness={args.plateau_sharpness}, "
          f"noise_level={args.noise_level} ({args.noise_mode}), scale2={args.scale2}")

    inits = make_shared_inits(oracle, args.n_campaigns, args.n_init, rng_seed=args.seed)

    results = {cond: [] for cond in CONDITIONS}
    for cond_name, (fn, kwargs) in CONDITIONS.items():
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
            "plateau_sharpness": args.plateau_sharpness, "noise_level": args.noise_level,
            "noise_mode": args.noise_mode, "scale2": args.scale2,
            "objective_names": oracle.objective_names(),
            "objective_directions": oracle.objective_directions(),
            "baseline_condition": BASELINE,
            "per_campaign_final_hv": results,
        }, f, indent=2)
    print(f"\nSaved per-campaign results to {args.out_path}")


if __name__ == "__main__":
    main()
