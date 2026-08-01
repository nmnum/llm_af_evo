"""
run_da_coreg_no_qnehvi_pilot.py — DA-COREG tested with qLogNEHVI removed
from the pipeline entirely, not just from the surrogate.

run_da_coreg_pilot.py's own DTLZ2 positive-control result (baseline_da_coreg:
-3.8%, 0/20 wins, p=1.9e-6) can't distinguish two different explanations:
  (a) DA-COREG's coregionalized posterior is bad, full stop, or
  (b) DA-COREG's posterior is fine, but qLogNEHVI's correlated joint MC
      batch scoring specifically compresses acquisition-value discrimination
      when fed a MultiTaskGP posterior (the leading, unproven hypothesis in
      L_COATINGS_FINDINGS.md Part 8).
That ablation kept qLogNEHVI-optimized qbo_x in the candidate pool and used
qLogNEHVI itself for scoring in EVERY cell (deliberately — to hold candidate
distribution fixed within THAT 2x2), so it never tested DA-COREG in a
pipeline qLogNEHVI never touches.

This script does: both conditions use full_replay.strategy_unsga3_pool_af
(UNSGA3-only candidate generation, trust_only score_pool scoring, top-k
select — no optimize_acqf/qLogNEHVI call anywhere), differing ONLY in
use_da_coreg. If unsga3_pool_af_da_coreg lands near unsga3_pool_af_indep
instead of repeating the -3.8%/p=1.9e-6 loss, that's evidence for (b); if it
loses just as badly here too, that's evidence for (a).

Usage:
    python run_da_coreg_no_qnehvi_pilot.py --n_replicates 1 --n_campaigns 3   # timing check
    python run_da_coreg_no_qnehvi_pilot.py --n_replicates 3 --n_campaigns 20  # real run
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
from full_replay import strategy_unsga3_pool_af
from af_interface import SEED_PROGRAMS
from synthetic_mo_oracle import DiscreteSyntheticMOOracle
from ada_coatings_oracle import DiscreteADACoatingsOracle
from excipient_oracle_mo import MultiObjectiveExcipientOracle

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"
TRUST_ONLY = SEED_PROGRAMS["trust_only"]

CONDITIONS = {
    "unsga3_pool_af_indep": (strategy_unsga3_pool_af,
                              {"af_code": TRUST_ONLY, "use_da_coreg": False}),
    "unsga3_pool_af_da_coreg": (strategy_unsga3_pool_af,
                                 {"af_code": TRUST_ONLY, "use_da_coreg": True}),
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
    # Same reasoning as run_da_coreg_pilot.py's n_fallback tracking: a
    # campaign that silently fell back to strategy_mo_egbo (e.g. from a
    # MultiTaskGP fit failure) must not look identical to a genuine
    # DA-COREG loss in the final_hv-only summary.
    n_fallback = sum(1 for d in result["decisions"] if d.get("unsga3_pool_af") is False)
    n_total = len(result["decisions"])
    fallback_reasons = [d.get("fallback_reason") for d in result["decisions"]
                         if d.get("unsga3_pool_af") is False]
    return final_hv, n_fallback, n_total, fallback_reasons


def run_one_replicate(oracle, n_campaigns, budget, n_init, batch_size, replicate_idx,
                       base_seed, n_fitness_seeds: int = 1):
    """
    n_fitness_seeds > 1 reruns each campaign (same X_init/Y_init) under
    multiple seeds and averages final_hv before it ever reaches the
    win/diff/Wilcoxon computation — same rationale as
    mab_noise_diagnostic.py: campaign-level seed noise (GP-fit + UNSGA3
    seed both derive from the single `seed` int passed to run_mo_campaign
    here) does not cancel by pairing conditions on the same seed, because
    the two conditions' trajectories diverge after batch 1 and decorrelate
    from there. Each extra seed multiplies wall-clock cost linearly — this
    does not decouple the GP-fit-seed vs UNSGA3-seed axes the way that
    diagnostic did, it just averages over the combined noise both
    contribute, which is enough to tell "ties baseline" from "small real
    effect" apart without the full two-axis decomposition.
    """
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
                # seed_offset + i selects the campaign's init; the *1000
                # + s term below is the ADDITIONAL fitness-seed axis, kept
                # far enough from the per-campaign stride (i only ranges
                # over n_campaigns, always << 1000) that seeds never
                # collide across campaigns or fitness-seed replicates.
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
    ap.add_argument("--domain", choices=["mab", "coatings", "dtlz2", "zdt1"], default="dtlz2")
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--replicate_start", type=int, default=0)
    ap.add_argument("--n_fitness_seeds", type=int, default=1,
                     help="Average each campaign's final_hv over this many seeds "
                          "before computing wins/diffs (cost multiplies linearly). "
                          "Use >1 when a single-seed run's win rate/p-value looks "
                          "noise-dominated rather than clean — see mab_noise_"
                          "diagnostic.py for the precedent.")
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    out_path = args.out_path or str(HERE / f"da_coreg_no_qnehvi_{args.domain}_results.json")

    oracle = build_oracle(args.domain)
    print(f"Domain: {args.domain} — {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print("Both conditions: UNSGA3-only candidates, trust_only score_pool scoring, "
          "top-k select — no qLogNEHVI/optimize_acqf call anywhere. Only the "
          "surrogate (independent GPs vs DA-COREG) differs.\n")
    print(f"Running {args.n_replicates} independent replicates of "
          f"{args.n_campaigns} campaigns x 2 conditions x {args.n_fitness_seeds} "
          f"fitness seed(s) each "
          f"(replicate_idx {args.replicate_start}..{args.replicate_start + args.n_replicates - 1})...\n")

    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small. Use this run for a timing/smoke check only.\n")

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
        # :.4g, not :.4f — see run_da_coreg_pilot.py's identical comment:
        # small p-values (e.g. ~1.9e-6 at n=20) print as a misleading
        # "0.0000" under :.4f instead of their actual scientific-notation
        # value.
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
    n_sig_positive = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] > 0)
    n_sig_negative = sum(1 for i in range(len(pcts)) if ps[i] < 0.05 and pcts[i] < 0)
    n_positive = sum(1 for p in pcts if p > 0)

    print(f"\n{'='*60}")
    print(f"ACROSS-REPLICATE PATTERN ({args.domain}, no-qLogNEHVI pipeline):")
    print(f"  % diff per replicate: {[f'{p:+.1f}%' for p in pcts]}")
    print(f"  wins per replicate:   {wins}")
    print(f"  p per replicate:      {[f'{p:.3g}' for p in ps]}")
    print(f"  mean % diff across replicates: {np.mean(pcts):+.1f}% (std {np.std(pcts):.1f})")
    print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
    print(f"  significant (p<0.05) POSITIVE in {n_sig_positive}/{len(pcts)} replicates")
    print(f"  significant (p<0.05) NEGATIVE in {n_sig_negative}/{len(pcts)} replicates")
    print(f"\nCompare to run_da_coreg_pilot.py's --domain dtlz2 result "
          f"(qLogNEHVI-in-the-loop): -3.8%, 0/20 wins, p=1.9e-6.")
    print("If this run's numbers are close to that, DA-COREG's posterior is bad on "
          "its own (hypothesis a). If this run ties/beats baseline instead, the "
          "loss was specific to qLogNEHVI's joint MC batch scoring (hypothesis b).")

    with open(out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {out_path}")


if __name__ == "__main__":
    main()
