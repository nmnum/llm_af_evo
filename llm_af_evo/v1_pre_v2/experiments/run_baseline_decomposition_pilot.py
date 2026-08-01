"""
run_baseline_decomposition_pilot.py — decomposes mo_egbo_novelty ITSELF,
the same discipline the eight-gate arc applied to L's evolved AFs (see
baseline_decomposition_strategies.py's module docstring for the full
rationale). Six conditions against the full baseline composite:

  mo_egbo_novelty     — the full composite (qLogNEHVI + UNSGA3 + novelty selection)
  strategy_mo_egbo_real — qLogNEHVI + UNSGA3 + top-k (isolates novelty selection's
                          own contribution; pre-existing, now direction-fixed)
  qnehvi_plain        — qLogNEHVI ALONE (isolates UNSGA3 + novelty selection's
                          combined contribution)
  unsga3_plain        — pure evolutionary, no GP/acquisition at all
  parego              — Chebyshev-scalarised UCB, the standard weak MOBO baseline
  random              — required floor

No new sandboxed code, no LLM calls — existing/standard strategy
implementations through the existing run_mo_campaign harness, on domains
this project already has data pipelines for (mAb, coatings — the same two
domains the whole eight-gate arc was run on).

Usage:
    python run_baseline_decomposition_pilot.py --domain mab --n_replicates 3 --n_campaigns 20
    python run_baseline_decomposition_pilot.py --domain coatings --n_replicates 3 --n_campaigns 20
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
from excipient_campaign_mo import (
    run_mo_campaign, make_shared_inits, strategy_mo_egbo_real, strategy_mo_random,
)
from excipient_oracle_mo import MultiObjectiveExcipientOracle
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from ada_coatings_oracle import DiscreteADACoatingsOracle
from baseline_decomposition_strategies import (
    strategy_qnehvi_plain, strategy_unsga3_plain, strategy_parego,
)

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"

CONDITIONS = {
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "mo_egbo_real": (strategy_mo_egbo_real, {}),
    "qnehvi_plain": (strategy_qnehvi_plain, {}),
    "unsga3_plain": (strategy_unsga3_plain, {}),
    "parego": (strategy_parego, {}),
    "random": (strategy_mo_random, {}),
}


def build_oracle(domain: str):
    if domain == "mab":
        full = MultiObjectiveExcipientOracle(
            protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
        return full.make_discrete_oracle(n_samples=500, seed=42)
    if domain == "coatings":
        return DiscreteADACoatingsOracle.build()
    raise ValueError(f"unknown domain {domain!r}")


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def run_one_replicate(oracle, n_campaigns, budget, n_init, batch_size, replicate_idx,
                       base_seed):
    seed_offset = base_seed + replicate_idx * 1000
    inits = make_shared_inits(oracle, n_campaigns, n_init, rng_seed=seed_offset)

    results = {cond: [] for cond in CONDITIONS}
    for cond_name, (fn, kwargs) in CONDITIONS.items():
        for i, (X_init, Y_init) in enumerate(inits):
            final_hv = run_one(oracle, X_init, Y_init, budget, batch_size,
                                seed_offset + i, fn, kwargs)
            results[cond_name].append(final_hv)
    return results


def summarize(results, n_campaigns):
    baseline = np.array(results["mo_egbo_novelty"])
    out = {}
    for cond_name in [c for c in CONDITIONS if c != "mo_egbo_novelty"]:
        af_hv = np.array(results[cond_name])
        diffs = af_hv - baseline
        pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        out[cond_name] = {"pct_diff": float(pct), "n_wins": n_wins,
                           "n_campaigns": n_campaigns, "p": float(pval)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["mab", "coatings"], default="mab")
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    out_path = args.out_path or str(HERE / f"baseline_decomposition_{args.domain}_results.json")

    oracle = build_oracle(args.domain)
    print(f"Domain: {args.domain} — {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print(f"Conditions: {list(CONDITIONS)}")
    print(f"Running {args.n_replicates} independent replicates of "
          f"{args.n_campaigns} campaigns x {len(CONDITIONS)} conditions...\n")

    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small. Use this run for a timing/smoke check only.\n")

    all_replicates = []
    for r in range(args.n_replicates):
        t0 = time.perf_counter()
        results = run_one_replicate(oracle, args.n_campaigns, args.budget,
                                     args.n_init, args.batch_size, r, args.base_seed)
        summary = summarize(results, args.n_campaigns)
        elapsed = time.perf_counter() - t0
        print(f"Replicate {r} ({elapsed:.0f}s):")
        for cond_name, s in summary.items():
            print(f"  {cond_name:<14} diff={s['pct_diff']:+.1f}%  "
                  f"wins={s['n_wins']}/{s['n_campaigns']}  p={s['p']:.4g}")
        all_replicates.append({"replicate": r, "per_campaign_final_hv": results,
                                "summary": summary})

    print(f"\n{'='*70}")
    print(f"ACROSS-REPLICATE PATTERN ({args.domain}):")
    for cond_name in [c for c in CONDITIONS if c != "mo_egbo_novelty"]:
        pcts = [rep["summary"][cond_name]["pct_diff"] for rep in all_replicates]
        wins = [rep["summary"][cond_name]["n_wins"] for rep in all_replicates]
        ps = [rep["summary"][cond_name]["p"] for rep in all_replicates]
        n_sig_positive = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] > 0)
        n_sig_negative = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] < 0)
        n_positive = sum(1 for p in pcts if p > 0)
        print(f"\n{cond_name}:")
        print(f"  % diff per replicate: {[f'{p:+.1f}%' for p in pcts]}")
        print(f"  wins per replicate:   {wins}")
        print(f"  p per replicate:      {[f'{p:.4g}' for p in ps]}")
        print(f"  mean % diff across replicates: {np.mean(pcts):+.1f}% (std {np.std(pcts):.1f})")
        print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) POSITIVE in {n_sig_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) NEGATIVE in {n_sig_negative}/{len(pcts)} replicates")

    print(f"\n{'='*70}")
    print("READ: qnehvi_plain tells you whether UNSGA3+novelty-selection add "
          "anything beyond plain joint MC acquisition in this regime. "
          "mo_egbo_real tells you novelty selection's own contribution "
          "specifically. unsga3_plain/parego/random are sanity floors — if "
          "parego or unsga3_plain also tie mo_egbo_novelty, this domain/budget "
          "may not discriminate between MOBO methods at all.")

    with open(out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {out_path}")


if __name__ == "__main__":
    main()
