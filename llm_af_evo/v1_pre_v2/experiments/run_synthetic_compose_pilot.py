"""
run_synthetic_compose_pilot.py — Step 3 of the compose_batch + DA-COREG
plan: the full 2x2 ablation (surrogate: independent GPs vs. DA-COREG x
selection: baseline novelty-weighted vs. compose_batch) run on synthetic
domains (ZDT1, DTLZ2) FIRST, before spending real-domain compute — this
shakes out compose_sandbox.py's new failure modes (timeout from the
greedy marginal-HV seed's per-step HV estimation, DA-COREG's MultiTaskGP
API assumptions) cheaply, with known Pareto-front structure giving a
gradient signal the real domains (mAb, coatings) structurally lack.

Four conditions, run as the full ablation from the start (not combined-
then-backfilled) — see compose_strategies.py's module docstring for why
each cell needs to be interpretable on its own:
  mo_egbo_novelty       — indep. GPs + baseline selection (existing, unmodified)
  baseline_da_coreg     — DA-COREG surrogate + baseline selection
  compose_indep         — indep. GPs + compose_batch (greedy_marginal_hv seed)
  compose_da_coreg      — DA-COREG surrogate + compose_batch ("combined")

Decision gate (pre-committed, per the plan's Step 3):
  combined wins, both single-lever cells tie baseline -> superadditive interaction
  combined wins, baseline_da_coreg also wins comparably -> surrogate-driven, not compose_batch
  combined wins, compose_indep also wins comparably -> generation/composition-driven, DA-COREG unnecessary
  nothing wins -> consistent negative, strengthens the decomposition-constraint conclusion

Usage:
    python run_synthetic_compose_pilot.py --domain zdt1 --n_replicates 1 --n_campaigns 3   # timing check
    python run_synthetic_compose_pilot.py --domain zdt1 --n_replicates 3 --n_campaigns 20  # real run
    python run_synthetic_compose_pilot.py --domain dtlz2 --n_replicates 3 --n_campaigns 20
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

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from compose_strategies import strategy_ablation_cell
from compose_interface import COMPOSE_SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent

CONDITIONS = {
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "baseline_da_coreg": (strategy_ablation_cell,
                           {"use_da_coreg": True, "use_compose": False}),
    "compose_indep": (strategy_ablation_cell,
                       {"use_da_coreg": False, "use_compose": True,
                        "compose_code": COMPOSE_SEED_PROGRAMS["greedy_marginal_hv"]}),
    "compose_da_coreg": (strategy_ablation_cell,
                          {"use_da_coreg": True, "use_compose": True,
                           "compose_code": COMPOSE_SEED_PROGRAMS["greedy_marginal_hv"]}),
    # Uncertainty-aware follow-up (see compose_interface.py's
    # SEED_GREEDY_MARGINAL_HV_MC docstring): combines compose_batch's
    # joint composition with mc_hvi_approx's MC-integration-over-
    # uncertainty, since greedy_marginal_hv alone LOST to baseline
    # (apparently from being mean-only/uncertainty-blind) and
    # mc_hvi_approx alone tied baseline as an independent score — tests
    # whether pairing the two recovers what neither did alone.
    "compose_mc_indep": (strategy_ablation_cell,
                          {"use_da_coreg": False, "use_compose": True,
                           "compose_code": COMPOSE_SEED_PROGRAMS["greedy_marginal_hv_mc"]}),
    "compose_mc_da_coreg": (strategy_ablation_cell,
                             {"use_da_coreg": True, "use_compose": True,
                              "compose_code": COMPOSE_SEED_PROGRAMS["greedy_marginal_hv_mc"]}),
}


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn is strategy_ablation_cell:
        kwargs["budget"] = budget
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
    ap.add_argument("--domain", choices=["zdt1", "dtlz2"], default="zdt1")
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    out_path = args.out_path or str(HERE / f"synthetic_compose_{args.domain}_results.json")

    oracle = (DiscreteSyntheticMOOracle.build_zdt1() if args.domain == "zdt1"
              else DiscreteSyntheticMOOracle.build_dtlz2())
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
            print(f"  {cond_name:<20} diff={s['pct_diff']:+.1f}%  "
                  f"wins={s['n_wins']}/{s['n_campaigns']}  p={s['p']:.4f}")
        all_replicates.append({"replicate": r, "per_campaign_final_hv": results,
                                "summary": summary})

    print(f"\n{'='*70}")
    print("ACROSS-REPLICATE PATTERN:")
    all_keys = list(all_replicates[0]["summary"].keys())
    for cond_name in all_keys:
        pcts = [rep["summary"][cond_name]["pct_diff"] for rep in all_replicates]
        wins = [rep["summary"][cond_name]["n_wins"] for rep in all_replicates]
        ps = [rep["summary"][cond_name]["p"] for rep in all_replicates]
        n_sig_positive = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] > 0)
        n_sig_negative = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] < 0)
        n_positive = sum(1 for p in pcts if p > 0)
        print(f"\n{cond_name}:")
        print(f"  % diff per replicate: {[f'{p:+.1f}%' for p in pcts]}")
        print(f"  wins per replicate:   {wins}")
        print(f"  p per replicate:      {[f'{p:.3f}' for p in ps]}")
        print(f"  mean % diff across replicates: {np.mean(pcts):+.1f}% "
              f"(std {np.std(pcts):.1f})")
        print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) POSITIVE in {n_sig_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) NEGATIVE in {n_sig_negative}/{len(pcts)} replicates")

    print(f"\n{'='*70}")
    print("DECISION-GATE READ:")
    for cond_name in ["compose_da_coreg", "baseline_da_coreg", "compose_indep",
                       "compose_mc_indep", "compose_mc_da_coreg"]:
        if cond_name not in all_replicates[0]["summary"]:
            continue
        pcts = [rep["summary"][cond_name]["pct_diff"] for rep in all_replicates]
        print(f"  {cond_name:<20} vs baseline: {[f'{p:+.1f}%' for p in pcts]}")
    print(f"  Compare compose_mc_* against compose_indep/compose_da_coreg: does adding "
          f"MC-integration-over-uncertainty recover what mean-only greedy composition "
          f"lost, per the module docstring's four interpretations.")

    with open(out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {out_path}")


if __name__ == "__main__":
    main()
