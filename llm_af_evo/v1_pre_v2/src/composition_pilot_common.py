"""
composition_pilot_common.py — shared replicate-runner/summary/reporting logic
for the "Within-Batch Joint Composition Pilot" (approach K gate), factored
out so run_composition_pilot.py (mAb) and run_composition_pilot_coatings.py
(ADA coatings) stay thin, domain-specific entry points instead of duplicating
the replicate/statistics machinery.

Callers must set the BLAS env vars (OMP_NUM_THREADS etc.) BEFORE importing
this module, exactly as before — this module itself imports numpy/scipy/
torch, so it's too late to set them here.
"""

import json
import time

import numpy as np
import torch
from scipy.stats import wilcoxon


def build_conditions(strategy_mo_egbo_novelty, strategy_evolved_af,
                      strategy_evolved_af_novelty, seed_programs):
    return {
        "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
        "trust_only_topk": (strategy_evolved_af, {"af_code": seed_programs["trust_only"]}),
        "trust_only_novelty": (strategy_evolved_af_novelty,
                                {"af_code": seed_programs["trust_only"]}),
        "ehvi_approx_topk": (strategy_evolved_af, {"af_code": seed_programs["ehvi_approx"]}),
        "ehvi_approx_novelty": (strategy_evolved_af_novelty,
                                 {"af_code": seed_programs["ehvi_approx"]}),
    }


def run_one(run_mo_campaign, strategy_evolved_af, strategy_evolved_af_novelty,
            oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn in (strategy_evolved_af, strategy_evolved_af_novelty):
        kwargs["budget"] = budget
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def run_one_replicate(conditions, run_mo_campaign, make_shared_inits,
                       strategy_evolved_af, strategy_evolved_af_novelty,
                       oracle, n_campaigns, budget, n_init, batch_size,
                       replicate_idx, base_seed):
    seed_offset = base_seed + replicate_idx * 1000
    inits = make_shared_inits(oracle, n_campaigns, n_init, rng_seed=seed_offset)

    results = {cond: [] for cond in conditions}
    for cond_name, (fn, kwargs) in conditions.items():
        for i, (X_init, Y_init) in enumerate(inits):
            final_hv = run_one(run_mo_campaign, strategy_evolved_af,
                                strategy_evolved_af_novelty, oracle, X_init, Y_init,
                                budget, batch_size, seed_offset + i, fn, kwargs)
            results[cond_name].append(final_hv)
    return results


def summarize(conditions, results, n_campaigns):
    baseline = np.array(results["mo_egbo_novelty"])
    out = {}
    for cond_name in [c for c in conditions if c != "mo_egbo_novelty"]:
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
    for base_name in ["trust_only", "ehvi_approx"]:
        topk = np.array(results[f"{base_name}_topk"])
        nov = np.array(results[f"{base_name}_novelty"])
        diffs = nov - topk
        pct = 100 * (nov.mean() - topk.mean()) / topk.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        out[f"{base_name}_novelty_vs_topk"] = {
            "pct_diff": float(pct), "n_wins": n_wins,
            "n_campaigns": n_campaigns, "p": float(pval)}
    return out


def run_pilot(conditions, run_mo_campaign, make_shared_inits, strategy_evolved_af,
              strategy_evolved_af_novelty, oracle, args, out_path):
    print(f"Conditions: {list(conditions)}")
    print(f"Running {args.n_replicates} independent replicates of "
          f"{args.n_campaigns} campaigns x {len(conditions)} conditions...\n")

    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small (max achievable two-sided p at n=3 is 0.25). Use this run for "
              f"a timing/smoke check only.\n")

    all_replicates = []
    for r in range(args.n_replicates):
        t0 = time.perf_counter()
        results = run_one_replicate(conditions, run_mo_campaign, make_shared_inits,
                                     strategy_evolved_af, strategy_evolved_af_novelty,
                                     oracle, args.n_campaigns, args.budget, args.n_init,
                                     args.batch_size, r, args.base_seed)
        summary = summarize(conditions, results, args.n_campaigns)
        elapsed = time.perf_counter() - t0
        print(f"Replicate {r} ({elapsed:.0f}s):")
        for cond_name, s in summary.items():
            print(f"  {cond_name:<28} diff={s['pct_diff']:+.1f}%  "
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
        if n_positive == len(pcts) or n_positive == 0:
            print(f"  => CONSISTENT direction across all replicates.")
        else:
            print(f"  => INCONSISTENT direction across replicates.")

    print(f"\n{'='*70}")
    print("GATE READ: for each scoring function, does novelty selection beat "
          "top-k, AND does the novelty-selection condition match/beat baseline?")
    for base_name in ["trust_only", "ehvi_approx"]:
        gate_key = f"{base_name}_novelty_vs_topk"
        nov_key = f"{base_name}_novelty"
        gate_pcts = [rep["summary"][gate_key]["pct_diff"] for rep in all_replicates]
        vs_base_pcts = [rep["summary"][nov_key]["pct_diff"] for rep in all_replicates]
        print(f"\n{base_name}: novelty vs topk = {[f'{p:+.1f}%' for p in gate_pcts]}  "
              f"| novelty vs baseline = {[f'{p:+.1f}%' for p in vs_base_pcts]}")

    with open(out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {out_path}")
