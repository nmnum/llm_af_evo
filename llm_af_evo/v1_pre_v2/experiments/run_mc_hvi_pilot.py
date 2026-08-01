"""
run_mc_hvi_pilot.py — the noise-integration gate, the next mechanism probe
after the batch-size ablation ruled OUT joint-batch-redundancy accounting
as the explanation for trust_only/ehvi_approx's uniform tie against
mo_egbo_novelty (no gap opened up at any batch size, including q=1, where
that mechanism should have been most visible).

qLogNEHVI is *noisy* EHVI — it Monte-Carlo integrates over the joint
posterior, which lets it properly weigh how trustworthy each point's
estimated position relative to the Pareto front is, given observation
noise. Neither trust_only (mean-sum only) nor ehvi_approx (single mean-
only point estimate, no sampling) does anything like this. mc_hvi_approx
(af_interface.SEED_PROGRAMS["mc_hvi_approx"]) is a numpy-only, per-
candidate decomposition of that idea: draw n_samples from each candidate's
posterior N(mean, std) and average the per-sample HV-improvement proxy —
faithful to the "integrate over posterior uncertainty" part of qLogNEHVI,
explicitly NOT faithful to its joint multi-candidate part (already ruled
out separately).

Conditions, all with top-k selection (batch_size=5, matching every prior
pilot's default so results are comparable):
  mo_egbo_novelty   — baseline (unmodified)
  trust_only_topk   — mean-only exploitation (reference, from the original L result)
  ehvi_approx_topk  — mean-only front-based proxy (reference)
  mc_hvi_approx_topk — the new noise-aware MC-integration proposer (the actual test)

Decision gate (pre-committed): if mc_hvi_approx_topk consistently
outperforms trust_only_topk/ehvi_approx_topk AND closes some/all of the
gap to mo_egbo_novelty across 3 replicates, that isolates noise-aware
scoring as the missing ingredient mean-only per-candidate scorers lack —
independent of, and additive to, the two already-ruled-out surfaces
(selection, generation) and the already-ruled-out batch-composition
mechanism. A null here would mean none of scoring/selection/generation/
batch-composition/noise-integration individually explains the tie —
narrowing what's left to some genuinely inseparable combination of these,
or a mechanism not yet tested.

Usage:
    python run_mc_hvi_pilot.py --n_replicates 3 --n_campaigns 20
    python run_mc_hvi_pilot.py --n_replicates 1 --n_campaigns 3   # timing smoke test
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
from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty
from full_replay import strategy_evolved_af
from af_interface import SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent
PROTEIN = "mAb_aggregation"

CONDITIONS = {
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "trust_only_topk": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["trust_only"]}),
    "ehvi_approx_topk": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["ehvi_approx"]}),
    "mc_hvi_approx_topk": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["mc_hvi_approx"]}),
}


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn is strategy_evolved_af:
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
    # The actual gate question: does mc_hvi_approx beat the mean-only seeds,
    # not just baseline?
    mc_hv = np.array(results["mc_hvi_approx_topk"])
    for ref_name in ["trust_only_topk", "ehvi_approx_topk"]:
        ref_hv = np.array(results[ref_name])
        diffs = mc_hv - ref_hv
        pct = 100 * (mc_hv.mean() - ref_hv.mean()) / ref_hv.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        out[f"mc_hvi_vs_{ref_name}"] = {"pct_diff": float(pct), "n_wins": n_wins,
                                         "n_campaigns": n_campaigns, "p": float(pval)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=str(HERE / "mc_hvi_pilot_results.json"))
    args = ap.parse_args()

    oracle_full = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
    oracle = oracle_full.make_discrete_oracle(n_samples=500, seed=42)
    print(f"Oracle: {len(oracle)} pool points, {oracle.objective_names()} "
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
    print("GATE READ: does mc_hvi_approx beat the mean-only seeds, and does "
          "that translate into closing the gap to baseline?")
    for ref_name in ["trust_only_topk", "ehvi_approx_topk"]:
        gate_key = f"mc_hvi_vs_{ref_name}"
        gate_pcts = [rep["summary"][gate_key]["pct_diff"] for rep in all_replicates]
        print(f"\nmc_hvi_approx vs {ref_name}: {[f'{p:+.1f}%' for p in gate_pcts]}")
    base_pcts = [rep["summary"]["mc_hvi_approx_topk"]["pct_diff"] for rep in all_replicates]
    print(f"\nmc_hvi_approx vs baseline: {[f'{p:+.1f}%' for p in base_pcts]}")

    with open(args.out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {args.out_path}")


if __name__ == "__main__":
    main()
