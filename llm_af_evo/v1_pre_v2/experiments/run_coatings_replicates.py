"""
run_coatings_replicates.py — run the full 20-campaign coatings comparison
N times as genuinely independent replicates (different init-point draws
AND different per-campaign seeds each time, not just re-running the same
nominal seed and hoping for determinism), and report whether the
trust_only/ehvi_approx-vs-baseline pattern is consistent across replicates.

This exists because run_coatings_generalization.py turned out not to be
bit-reproducible even with matched seeds and single-threaded BLAS (env
vars + torch.set_num_threads(1)) — the remaining sensitivity is most
likely candidates snapping to different real neighbours across tiny
optimizer jitter, on a 253-point real pool covering a 4D continuous
space (see the run-to-run diffs from the two prior sanity-check runs).
Rather than keep chasing bit-exact reproducibility, this measures the
run-to-run variance directly: does the SAME qualitative finding show up
across several independent draws, or was any single run's p-value a
coin flip?

Usage:
    python run_coatings_replicates.py --n_replicates 3 --n_campaigns 20
"""

import os

# Same reasoning as run_coatings_generalization.py's identical block: must
# be set before numpy/scipy/torch are imported ANYWHERE in this process,
# including by this file itself — importing run_coatings_generalization
# second (as this file originally did) is too late, since `import numpy`
# below would already have run first and initialized BLAS threading.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from run_coatings_generalization import CONDITIONS, run_one
from ada_coatings_oracle import DiscreteADACoatingsOracle
from excipient_campaign_mo import make_shared_inits

HERE = pathlib.Path(__file__).parent


def run_one_replicate(oracle, n_campaigns, budget, n_init, batch_size, replicate_idx,
                       base_seed):
    # Distinct seed block per replicate — both for which init points get
    # drawn (make_shared_inits' rng_seed) and for each campaign's own
    # torch/numpy seed (run_one's seed=) — so replicates are genuinely
    # independent draws, not the same nominal experiment repeated.
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_replicates", type=int, default=3)
    ap.add_argument("--n_campaigns", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--out_path", default=str(HERE / "coatings_replicates_results.json"))
    args = ap.parse_args()

    oracle = DiscreteADACoatingsOracle.build()
    print(f"Oracle: {len(oracle)} real samples, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")
    print(f"Running {args.n_replicates} independent replicates of "
          f"{args.n_campaigns} campaigns x 3 conditions...\n")

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
                  f"wins={s['n_wins']}/{s['n_campaigns']}  p={s['p']:.4f}")
        all_replicates.append({"replicate": r, "per_campaign_final_hv": results,
                                "summary": summary})

    print(f"\n{'='*60}")
    print("ACROSS-REPLICATE PATTERN:")
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
        print(f"  p per replicate:      {[f'{p:.3f}' for p in ps]}")
        print(f"  mean % diff across replicates: {np.mean(pcts):+.1f}% "
              f"(std {np.std(pcts):.1f})")
        print(f"  directionally positive in {n_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) POSITIVE in {n_sig_positive}/{len(pcts)} replicates")
        print(f"  significant (p<0.05) NEGATIVE in {n_sig_negative}/{len(pcts)} replicates")
        if n_positive == len(pcts) or n_positive == 0:
            print(f"  => CONSISTENT direction across all replicates.")
        else:
            print(f"  => INCONSISTENT direction across replicates — treat any single "
                  f"run's result as unreliable.")

    with open(args.out_path, "w") as f:
        json.dump(all_replicates, f, indent=2)
    print(f"\nSaved full replicate data to {args.out_path}")


if __name__ == "__main__":
    main()
