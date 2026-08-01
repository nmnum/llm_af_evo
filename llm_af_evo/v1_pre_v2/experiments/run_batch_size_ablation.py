"""
run_batch_size_ablation.py — the mechanism probe motivated by three
independent nulls (scoring: run_composition_pilot.py's original L result;
selection: run_composition_pilot.py's novelty-selection gate; generation:
run_generation_pilot.py's geometry-only gate). None of the three decomposed
pipeline stages individually explains or closes the gap to
qLogNEHVI+UNSGA3+novelty-selection — which points at the JOINT, q>1 batch
optimization itself as the binding constraint, not any one stage.

The direct test: qLogNEHVI's batch treatment collapses to plain (noisy)
EHVI at q=1 — there's no batch to jointly reason about, and "score each
candidate independently, pick the best" becomes exactly what a q=1
acquisition function already does. If the trust_only_topk-vs-baseline gap
SHRINKS toward 0 as batch_size drops from 5 toward 1, that's direct
evidence the missing ingredient is joint batch-redundancy accounting
specifically (candidates discounted for being correlated/redundant with
OTHER candidates in the same batch — a quantity no per-candidate score,
however well-informed, can express). If the gap PERSISTS at q=1, joint
batch composition isn't the mechanism and the explanation lies elsewhere
(e.g. qLogNEHVI's *noisy* variant properly integrating observation noise
into which points are trusted as Pareto-optimal, vs. our point-estimate
GP-mean scorers).

This also directly speaks to LLaMEA-BO (van Stein & Bäck 2025): it reports
LLM-rewritten-optimizer success only in a SEQUENTIAL q=1 setting. If our
own gap vanishes at q=1 too, that's not a coincident contradiction — it's
the same mechanism producing both results, and the honest scope of
"per-candidate/decomposed AF evolution works" narrows to strictly
sequential BO, which our short-horizon BATCH setting was never inside.

Cost warning: total sandbox/GP-fit calls per campaign scale as
(budget - n_init) / batch_size — batch_size=1 means ~5x more strategy
calls than batch_size=5 for the same budget. Defaults are deliberately
smaller (n_campaigns=10, n_replicates=1) than the other pilots' 20/3 —
run the timing smoke test at n_campaigns=3 first regardless, and scale up
only for the batch sizes that turn out affordable.

Usage:
    python run_batch_size_ablation.py --batch_sizes 1,2,3,5,10 --n_campaigns 3 --n_replicates 1   # timing check
    python run_batch_size_ablation.py --batch_sizes 1,2,3,5,10 --n_campaigns 10 --n_replicates 3  # real run
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
}


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)
    kwargs = dict(kwargs)
    if fn is strategy_evolved_af:
        kwargs["budget"] = budget
    result = run_mo_campaign(oracle, X_init, Y_init, budget, fn, kwargs,
                              batch_size=batch_size, seed=seed)
    return result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")


def run_one_batch_size(oracle, batch_size, n_campaigns, budget, n_init,
                        n_replicates, base_seed):
    replicate_summaries = []
    for r in range(n_replicates):
        seed_offset = base_seed + r * 1000
        inits = make_shared_inits(oracle, n_campaigns, n_init, rng_seed=seed_offset)

        results = {cond: [] for cond in CONDITIONS}
        t0 = time.perf_counter()
        for cond_name, (fn, kwargs) in CONDITIONS.items():
            for i, (X_init, Y_init) in enumerate(inits):
                final_hv = run_one(oracle, X_init, Y_init, budget, batch_size,
                                    seed_offset + i, fn, kwargs)
                results[cond_name].append(final_hv)
        elapsed = time.perf_counter() - t0

        baseline = np.array(results["mo_egbo_novelty"])
        af_hv = np.array(results["trust_only_topk"])
        diffs = af_hv - baseline
        pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        replicate_summaries.append({
            "pct_diff": float(pct), "n_wins": n_wins, "n_campaigns": n_campaigns,
            "p": float(pval), "elapsed_s": elapsed,
            "per_campaign_final_hv": results,
        })
    return replicate_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_sizes", default="1,2,3,5,10",
                     help="Comma-separated batch sizes to sweep, from smallest (closest "
                          "to sequential q=1) to largest.")
    ap.add_argument("--n_campaigns", type=int, default=10)
    ap.add_argument("--n_replicates", type=int, default=1)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=str(HERE / "batch_size_ablation_results.json"))
    args = ap.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    oracle_full = MultiObjectiveExcipientOracle(
        protein=PROTEIN, tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10, seed=42)
    oracle = oracle_full.make_discrete_oracle(n_samples=500, seed=42)
    print(f"Oracle: {len(oracle)} pool points, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print(f"Sweeping batch_size in {batch_sizes}, {args.n_replicates} replicate(s) x "
          f"{args.n_campaigns} campaigns each (budget={args.budget}, n_init={args.n_init})")
    print(f"NOTE: cost per batch_size scales as ~(budget-n_init)/batch_size — "
          f"batch_size={min(batch_sizes)} will take ~{max(batch_sizes)//min(batch_sizes)}x "
          f"longer per campaign than batch_size={max(batch_sizes)}.\n")

    all_results = {}
    for bs in batch_sizes:
        n_batches = max(1, (args.budget - args.n_init) // bs)
        print(f"batch_size={bs} ({n_batches} batches/campaign)...")
        reps = run_one_batch_size(oracle, bs, args.n_campaigns, args.budget,
                                   args.n_init, args.n_replicates, args.base_seed)
        for rep in reps:
            print(f"  diff={rep['pct_diff']:+.1f}%  wins={rep['n_wins']}/{rep['n_campaigns']}  "
                  f"p={rep['p']:.4f}  ({rep['elapsed_s']:.0f}s)")
        all_results[str(bs)] = reps

    print(f"\n{'='*70}")
    print("TREND ACROSS BATCH SIZE (trust_only_topk vs. mo_egbo_novelty):")
    print(f"{'batch_size':<12}{'mean diff':<14}{'std':<10}{'n_sig(any dir)':<16}")
    for bs in batch_sizes:
        reps = all_results[str(bs)]
        pcts = [r["pct_diff"] for r in reps]
        ps = [r["p"] for r in reps]
        n_sig = sum(1 for p in ps if p < 0.05)
        print(f"{bs:<12}{np.mean(pcts):+.1f}%{'':<8}{np.std(pcts):<10.1f}{n_sig:<16}")

    print(f"\nIf the gap (mean diff, currently negative/near-zero at batch_size=5 per "
          f"prior pilots) trends toward 0 or positive as batch_size decreases toward 1, "
          f"that supports the joint-batch-redundancy hypothesis. If it stays flat "
          f"regardless of batch_size, joint composition isn't the explanation.")

    with open(args.out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {args.out_path}")


if __name__ == "__main__":
    main()
