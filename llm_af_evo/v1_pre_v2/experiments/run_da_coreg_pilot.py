"""
run_da_coreg_pilot.py — DA-COREG tested alone, decoupled from compose_batch
entirely (per the decision to promote it from "equal billing" to primary
next step after compose_batch's own negative result — see
L_COATINGS_FINDINGS.md Part 7). Two conditions only:
  mo_egbo_novelty    — baseline, independent per-objective GPs (unmodified)
  baseline_da_coreg  — strategy_ablation_cell(use_da_coreg=True, use_compose=False):
                       identical candidate generation and identical baseline
                       selection (qLogNEHVI-score + novelty-weighted select),
                       the ONLY difference is the surrogate: a coregionalized
                       multi-task GP instead of independent per-objective GPs.

This is the SAME cell already validated clean on synthetic ZDT1 (ties
baseline, +0.1% mean, never significant) — this script runs it on the
domains that actually matter: real-domain data, and (recommended first,
free) DTLZ2 as a positive-control synthetic domain with engineered shared
cross-objective structure (all 3 objectives share a common g(x) term),
which the correlation check (check_objective_correlation.py) should be
run before this to set expectations per domain — DA-COREG has nothing to
exploit on a near-independent-objectives domain like coatings, regardless
of implementation quality.

Usage:
    python check_objective_correlation.py                              # run first, sets expectations
    python run_da_coreg_pilot.py --domain dtlz2 --n_replicates 3 --n_campaigns 20   # positive control
    python run_da_coreg_pilot.py --domain mab --n_replicates 3 --n_campaigns 20
    python run_da_coreg_pilot.py --domain coatings --n_replicates 3 --n_campaigns 20  # low prior, per correlation check
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
from excipient_oracle_mo import MultiObjectiveExcipientOracle
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from ada_coatings_oracle import DiscreteADACoatingsOracle
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from compose_strategies import strategy_ablation_cell

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"

CONDITIONS = {
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "baseline_da_coreg": (strategy_ablation_cell,
                           {"use_da_coreg": True, "use_compose": False}),
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
    if fn is strategy_ablation_cell:
        kwargs["budget"] = budget
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
    # Surface strategy_ablation_cell's own per-batch fallback flag (merged
    # into each decisions[b] dict via run_mo_campaign's **log_extra) —
    # without this, a campaign that silently fell back to the much weaker
    # strategy_mo_egbo on some/all batches (e.g. from a MultiTaskGP fit
    # failure) looks identical in the final_hv-only summary to a campaign
    # where DA-COREG genuinely ran and genuinely underperformed. Only
    # meaningful for fn is strategy_ablation_cell — mo_egbo_novelty's own
    # decisions dicts don't have this key, so n_fallback stays 0 for it.
    n_fallback = sum(1 for d in result["decisions"] if d.get("ablation_cell_ok") is False)
    n_total = len(result["decisions"])
    fallback_reasons = [d.get("fallback_reason") for d in result["decisions"]
                         if d.get("ablation_cell_ok") is False]
    return final_hv, n_fallback, n_total, fallback_reasons


def run_one_replicate(oracle, n_campaigns, budget, n_init, batch_size, replicate_idx,
                       base_seed):
    seed_offset = base_seed + replicate_idx * 1000
    inits = make_shared_inits(oracle, n_campaigns, n_init, rng_seed=seed_offset)

    results = {cond: [] for cond in CONDITIONS}
    fallback_info = {cond: [] for cond in CONDITIONS}
    for cond_name, (fn, kwargs) in CONDITIONS.items():
        for i, (X_init, Y_init) in enumerate(inits):
            final_hv, n_fallback, n_total, reasons = run_one(
                oracle, X_init, Y_init, budget, batch_size, seed_offset + i, fn, kwargs)
            results[cond_name].append(final_hv)
            fallback_info[cond_name].append(
                {"campaign": i, "n_fallback": n_fallback, "n_total": n_total,
                 "reasons": reasons})
    return results, fallback_info


def summarize(results, n_campaigns):
    baseline = np.array(results["mo_egbo_novelty"])
    af_hv = np.array(results["baseline_da_coreg"])
    diffs = af_hv - baseline
    pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
    n_wins = int(np.sum(diffs > 0))
    try:
        _, pval = wilcoxon(diffs)
    except ValueError:
        pval = 1.0
    return {"baseline_da_coreg": {"pct_diff": float(pct), "n_wins": n_wins,
                                   "n_campaigns": n_campaigns, "p": float(pval)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["mab", "coatings", "dtlz2", "zdt1"], default="dtlz2")
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--replicate_start", type=int, default=0,
                     help="Offset the replicate index (seed_offset = base_seed + "
                          "replicate_idx*1000) — set to 2 to re-probe exactly the "
                          "seed block a prior 3-replicate run's replicate 2 used, "
                          "without re-running replicates 0/1.")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    out_path = args.out_path or str(HERE / f"da_coreg_{args.domain}_results.json")

    oracle = build_oracle(args.domain)
    print(f"Domain: {args.domain} — {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print(f"Running {args.n_replicates} independent replicates of "
          f"{args.n_campaigns} campaigns x 2 conditions "
          f"(replicate_idx {args.replicate_start}..{args.replicate_start + args.n_replicates - 1})...\n")

    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small. Use this run for a timing/smoke check only.\n")

    all_replicates = []
    for r in range(args.replicate_start, args.replicate_start + args.n_replicates):
        t0 = time.perf_counter()
        results, fallback_info = run_one_replicate(
            oracle, args.n_campaigns, args.budget, args.n_init, args.batch_size,
            r, args.base_seed)
        summary = summarize(results, args.n_campaigns)
        elapsed = time.perf_counter() - t0
        s = summary["baseline_da_coreg"]
        da_fallback = fallback_info["baseline_da_coreg"]
        total_fallback_batches = sum(c["n_fallback"] for c in da_fallback)
        total_batches = sum(c["n_total"] for c in da_fallback)
        n_campaigns_with_any_fallback = sum(1 for c in da_fallback if c["n_fallback"] > 0)
        # :.4f silently prints "0.0000" for any p below 5e-5 (e.g. the
        # theoretical Wilcoxon minimum at n=20, ~1.9e-6, when every
        # campaign has the same-sign difference) — that reads as "exactly
        # zero" when it isn't. :.4g switches to scientific notation for
        # small values instead of truncating them to a misleading zero.
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

    pcts = [rep["summary"]["baseline_da_coreg"]["pct_diff"] for rep in all_replicates]
    wins = [rep["summary"]["baseline_da_coreg"]["n_wins"] for rep in all_replicates]
    ps = [rep["summary"]["baseline_da_coreg"]["p"] for rep in all_replicates]
    n_sig_positive = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] > 0)
    n_sig_negative = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] < 0)
    n_positive = sum(1 for p in pcts if p > 0)

    print(f"\n{'='*60}")
    print(f"ACROSS-REPLICATE PATTERN ({args.domain}):")
    print(f"  % diff per replicate: {[f'{p:+.1f}%' for p in pcts]}")
    print(f"  wins per replicate:   {wins}")
    print(f"  p per replicate:      {[f'{p:.3f}' for p in ps]}")
    print(f"  mean % diff across replicates: {np.mean(pcts):+.1f}% (std {np.std(pcts):.1f})")
    print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
    print(f"  significant (p<0.05) POSITIVE in {n_sig_positive}/{len(pcts)} replicates")
    print(f"  significant (p<0.05) NEGATIVE in {n_sig_negative}/{len(pcts)} replicates")

    with open(out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {out_path}")


if __name__ == "__main__":
    main()
