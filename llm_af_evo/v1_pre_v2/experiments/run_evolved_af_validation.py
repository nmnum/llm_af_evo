"""
run_evolved_af_validation.py — Gate 3 follow-up: does evolve_af.py's best
evolved AF (evolution_runs/gate3_coreg_mab/best_af.py, fitness=0.4656,
win_rate=0.531 on its per-step proxy) actually help in real closed-loop
campaigns, or was the fitness proxy not a good tell?

Why this is a separate question from the fitness reported during evolution:
evolve_af.py's fitness measures, per training step, whether the candidate
AF's single-batch pick beats the baseline's (strategy_ablation_cell,
use_da_coreg=True) pick on predicted HV gain at THAT step, using already-
observed real campaign trajectories as fixed context. It never lets the
candidate AF drive its own multi-batch closed-loop campaign — so a small
per-step edge (or noise in that per-step proxy) may not survive, or may
compound differently, once errors from batch i feed into what the AF sees
at batch i+1. This script runs the evolved AF (as a score_pool inside
strategy_unsga3_pool_af, same harness as the ParEGO-EI smoke tests) through
real closed-loop mAb campaigns end to end, and compares final hypervolume
against a simple, un-evolved baseline (trust_only) under the SAME DA-COREG
surrogate — isolating exactly what evolution changed.

Same two-condition structure as run_parego_ei_da_coreg_smoke.py
(unsga3_pool_af_indep vs unsga3_pool_af_da_coreg), only here BOTH conditions
use_da_coreg=True and af_code differs (EVOLVED_AF vs TRUST_ONLY_BASELINE) —
the comparison of interest is "did evolution's search find something better
than doing nothing clever," not surrogate choice (already covered by Gate 2).

Usage:
    python run_evolved_af_validation.py                                  # smoke default
    python run_evolved_af_validation.py --n_campaigns 20 --n_replicates 3 # real run
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

import torch
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from full_replay import strategy_unsga3_pool_af
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from ada_coatings_oracle import DiscreteADACoatingsOracle
from excipient_oracle_mo import MultiObjectiveExcipientOracle

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"

# evolve_af.py's best evolved AF (evolution_runs/gate3_coreg_mab/best_af.py,
# fitness=0.4656, win_rate=0.531 on the per-step proxy), verbatim.
EVOLVED_AF = '''
def score_pool(context):
    """Adaptive exploitation-uncertainty balance with novelty reward, using inverse uncertainty weighting and progressive exploration emphasis."""
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        sigma_inv_sum = (1.0 / (gp["Tm"]["std"] / front_range["Tm"] + 1e-8) +
                         1.0 / (gp["kD"]["std"] / front_range["kD"] + 1e-8) +
                         1.0 / (gp["viscosity"]["std"] / front_range["viscosity"] + 1e-8))
        # Progressive emphasis: early more uncertainty-based, later more exploitation-based
        weight = 0.2 + 0.8 * (1.0 - progress)
        scores.append(mu_sum * weight + sigma_inv_sum * (1.0 - weight) + 0.3 * np.linalg.norm(X_obs - cand["x"], axis=1).min())
    return scores
'''.strip("\n")

# Simple, un-evolved baseline (af_interface.SEED_TRUST_ONLY, verbatim) —
# pure exploitation, no uncertainty/novelty term at all. Comparing the
# evolved AF against this (both under DA-COREG) isolates what evolution's
# search actually added, independent of the surrogate-choice question
# already answered by Gate 2.
TRUST_ONLY_BASELINE = '''
def score_pool(context):
    """Pure exploitation: rank by predicted objective sum only."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        scores.append(sum(gp[name]["mean"] for name in names))
    return scores
'''.strip("\n")

CONDITIONS = {
    "unsga3_pool_af_indep": (strategy_unsga3_pool_af,
                              {"af_code": TRUST_ONLY_BASELINE, "use_da_coreg": True}),
    "unsga3_pool_af_da_coreg": (strategy_unsga3_pool_af,
                                 {"af_code": EVOLVED_AF, "use_da_coreg": True}),
}


def build_oracle(domain: str):
    if domain == "mab":
        full = MultiObjectiveExcipientOracle(
            protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
        return full.make_discrete_oracle(n_samples=500, seed=42)
    if domain == "coatings":
        return DiscreteADACoatingsOracle.build()
    if domain == "dtlz2":
        return DiscreteSyntheticMOOracle.build_dtlz2()
    if domain == "zdt1":
        return DiscreteSyntheticMOOracle.build_zdt1()
    raise ValueError(f"unknown domain {domain!r}")


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    kwargs["budget"] = budget
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
    n_fallback = sum(1 for d in result["decisions"] if d.get("unsga3_pool_af") is False)
    n_total = len(result["decisions"])
    fallback_reasons = [d.get("fallback_reason") for d in result["decisions"]
                         if d.get("unsga3_pool_af") is False]
    return final_hv, n_fallback, n_total, fallback_reasons


def run_one_replicate(oracle, n_campaigns, budget, n_init, batch_size, replicate_idx,
                       base_seed, n_fitness_seeds: int = 1):
    seed_offset = base_seed + replicate_idx * 1000
    inits = make_shared_inits(oracle, n_campaigns, n_init, rng_seed=seed_offset)

    results = {cond: [] for cond in CONDITIONS}
    fallback_info = {cond: [] for cond in CONDITIONS}
    for cond_name, (fn, kwargs) in CONDITIONS.items():
        for i, (X_init, Y_init) in enumerate(inits):
            seed_hvs = []
            n_fallback_total = n_total_total = 0
            reasons_total = []
            for s in range(n_fitness_seeds):
                seed = seed_offset + i + s * 100_000
                final_hv, n_fallback, n_total, reasons = run_one(
                    oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs)
                seed_hvs.append(final_hv)
                n_fallback_total += n_fallback
                n_total_total += n_total
                reasons_total.extend(reasons)
            results[cond_name].append(float(np.mean(seed_hvs)))
            fallback_info[cond_name].append(
                {"campaign": i, "n_fallback": n_fallback_total, "n_total": n_total_total,
                 "reasons": reasons_total, "per_seed_hv": seed_hvs})
    return results, fallback_info


def summarize(results, n_campaigns):
    baseline = np.array(results["unsga3_pool_af_indep"])
    af_hv = np.array(results["unsga3_pool_af_da_coreg"])
    diffs = af_hv - baseline
    pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
    n_wins = int(np.sum(diffs > 0))
    try:
        _, pval = wilcoxon(diffs)
    except ValueError:
        pval = 1.0
    return {"unsga3_pool_af_da_coreg": {"pct_diff": float(pct), "n_wins": n_wins,
                                         "n_campaigns": n_campaigns, "p": float(pval)}}


def main():
    ap = argparse.ArgumentParser()
    # EVOLVED_AF hardcodes "Tm"/"kD"/"viscosity" object names (it evolved
    # against mAb training data specifically) — mab only, not domain-generic.
    ap.add_argument("--domain", choices=["mab"], default="mab")
    ap.add_argument("--n_replicates", type=int, default=1)
    ap.add_argument("--n_campaigns", type=int, default=5)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--replicate_start", type=int, default=0)
    ap.add_argument("--n_fitness_seeds", type=int, default=1)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    out_path = args.out_path or str(HERE / f"evolved_af_validation_{args.domain}_results.json")

    oracle = build_oracle(args.domain)
    print(f"Domain: {args.domain} — {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print("Both conditions: UNSGA3-only candidates, DA-COREG surrogate, top-k "
          "select — no qLogNEHVI/optimize_acqf call anywhere. Only score_pool "
          "differs: trust_only baseline (pure exploitation, un-evolved) vs "
          "EVOLVED_AF (evolve_af.py's best output from gate3_coreg_mab, "
          "fitness=0.4656 win_rate=0.531 on the per-step proxy).\n")
    print(f"Running {args.n_replicates} replicate(s) of {args.n_campaigns} campaigns "
          f"x 2 conditions x {args.n_fitness_seeds} fitness seed(s) each "
          f"(replicate_idx {args.replicate_start}..{args.replicate_start + args.n_replicates - 1})...\n")

    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small. Smoke/timing check only (§22 step 2's stated purpose): confirm "
              f"it runs and look for a directional sign before any seed-averaging "
              f"spend.\n")

    all_replicates = []
    for r in range(args.replicate_start, args.replicate_start + args.n_replicates):
        t0 = time.perf_counter()
        results, fallback_info = run_one_replicate(
            oracle, args.n_campaigns, args.budget, args.n_init, args.batch_size,
            r, args.base_seed, n_fitness_seeds=args.n_fitness_seeds)
        summary = summarize(results, args.n_campaigns)
        elapsed = time.perf_counter() - t0
        s = summary["unsga3_pool_af_da_coreg"]
        da_fallback = fallback_info["unsga3_pool_af_da_coreg"]
        total_fallback_batches = sum(c["n_fallback"] for c in da_fallback)
        total_batches = sum(c["n_total"] for c in da_fallback)
        n_campaigns_with_any_fallback = sum(1 for c in da_fallback if c["n_fallback"] > 0)
        p_str = f"{s['p']:.4g}"
        print(f"Replicate {r} ({elapsed:.0f}s): diff={s['pct_diff']:+.1f}%  "
              f"wins={s['n_wins']}/{s['n_campaigns']}  p={p_str}  "
              f"| DA-COREG fallback: {total_fallback_batches}/{total_batches} batches, "
              f"{n_campaigns_with_any_fallback}/{args.n_campaigns} campaigns affected")
        if total_fallback_batches > 0:
            reasons = [r_ for c in da_fallback for r_ in c["reasons"]]
            print(f"  Sample fallback reasons: {reasons[:3]}")
        all_replicates.append({"replicate": r, "per_campaign_final_hv": results,
                                "summary": summary, "fallback_info": fallback_info})

    pcts = [rep["summary"]["unsga3_pool_af_da_coreg"]["pct_diff"] for rep in all_replicates]
    wins = [rep["summary"]["unsga3_pool_af_da_coreg"]["n_wins"] for rep in all_replicates]
    ps = [rep["summary"]["unsga3_pool_af_da_coreg"]["p"] for rep in all_replicates]
    n_positive = sum(1 for p in pcts if p > 0)

    print(f"\n{'='*60}")
    print(f"GATE 3 CLOSED-LOOP VALIDATION ({args.domain}, evolved AF vs trust_only, "
          f"both DA-COREG):")
    print(f"  % diff per replicate: {[f'{p:+.1f}%' for p in pcts]}")
    print(f"  wins per replicate:   {wins}")
    print(f"  p per replicate:      {[f'{p:.3g}' for p in ps]}")
    print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
    print("\nIf flat/negative here, the evolved AF's per-step fitness (0.4656/win_rate "
          "0.531) did not translate into real closed-loop campaign gains — consistent "
          "with the fitness proxy not being a good tell, or with the evolved AF simply "
          "not being better than doing nothing clever. If positive, worth a real "
          "seed-averaged run (--n_campaigns 20 --n_replicates 3) before drawing "
          "conclusions either way.")

    with open(out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {out_path}")


if __name__ == "__main__":
    main()
