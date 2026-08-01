"""
run_coatings_generalization.py — Step 2/3 of the coatings generalization
test: does the L-negative design rule ("closed-form per-candidate scoring
ties, doesn't beat, a jointly-optimized MC-integrated acquisition strategy
on short-horizon noisy MO landscapes") hold on MacLeod et al. 2022's real
ADA coatings dataset, not just the mAb formulation domain?

Three conditions, run directly on the live DiscreteADACoatingsOracle (no
"reconstruct from a logged campaign" indirection needed — unlike the mAb
held-out validation, we have the real oracle in hand, not a snapshot of it):
  mo_egbo_novelty  — the baseline (strategy_mo_egbo_novelty, unmodified)
  trust_only       — pure exploitation (parameterized, regression-tested)
  ehvi_approx      — front-based HV-improvement proxy (parameterized, tested)

Defaults to a SMALL PILOT (3 campaigns) first — per-batch cost on a 4D/
253-point oracle hasn't been measured and every transplanted-from-mAb
timing estimate in this project has been wrong so far. Check the printed
per-campaign time before scaling to --n_campaigns 20.

Usage:
    python run_coatings_generalization.py --n_campaigns 3    # pilot
    python run_coatings_generalization.py --n_campaigns 20   # full run, once pilot timing looks OK
"""

import os

# MUST be set before numpy/scipy/torch are imported anywhere in the process
# — these libraries read BLAS thread-count env vars once at their own C-
# extension init time, so setting them later (e.g. via torch.set_num_threads
# after `import torch`) cannot retroactively fix a BLAS backend that's
# already initialized. Verified this is necessary, not just torch's own
# threading: even the unmodified baseline strategy (no evolved-AF code
# involved at all) produced different final_hv across two "same seed" runs
# after torch.set_num_threads(1) alone — the remaining non-determinism is
# in scipy's L-BFGS-B (used inside optimize_acqf) and/or numpy's BLAS
# backend, both of which are env-var-controlled, not torch-API-controlled.
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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from excipient_campaign_mo import run_mo_campaign, make_shared_inits
from strategy_ls_na_egbo import strategy_mo_egbo_novelty

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import torch
from ada_coatings_oracle import DiscreteADACoatingsOracle
from full_replay import strategy_evolved_af
from af_interface import SEED_PROGRAMS

HERE = pathlib.Path(__file__).parent

# The actual LLM-evolved winner from evolve_af.py's mAb run (evolution_runs/
# run1/best_af.py) — a genuine adaptive strategy (uncertainty weighted
# MORE as progress increases, exploitation weighted LESS), not a hand-
# designed guess. Parameterized here the same way trust_only/ehvi_approx
# were (loop over objective_names, no re-flip since gp[name]["mean"]/
# ["std"] are already all-maximise convention) — the original hardcodes
# Tm/kD/viscosity and would KeyError on coatings' 2-objective set otherwise.
_EVOLVED_ADAPTIVE_MAB_WINNER = '''
def score_pool(context):
    names = context["objective_names"]
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        uncertainty = sum(gp[name]["std"] / front_range[name] for name in names)
        novelty = np.linalg.norm(X_obs - cand["x"], axis=1).min()
        exploitation = sum(gp[name]["mean"] for name in names) / len(names)
        scores.append(uncertainty * (2.0 + progress) + novelty * 0.1
                       + exploitation * (1.0 - progress))
    return scores
'''.strip("\n")

CONDITIONS = {
    "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}),
    "trust_only": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["trust_only"]}),
    "ehvi_approx": (strategy_evolved_af, {"af_code": SEED_PROGRAMS["ehvi_approx"]}),
    "phase_decaying_ucb": (strategy_evolved_af,
                            {"af_code": SEED_PROGRAMS["phase_decaying_ucb"]}),
    "evolved_adaptive": (strategy_evolved_af,
                          {"af_code": _EVOLVED_ADAPTIVE_MAB_WINNER}),
}


def run_one(oracle, X_init, Y_init, budget, batch_size, seed, fn, kwargs):
    torch.manual_seed(seed)  # same reproducibility fix as full_replay.py
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
    ap.add_argument("--out_path", default=str(HERE / "coatings_generalization_results.json"),
                     help="Per-campaign, per-condition final_hv, saved so later analysis "
                          "(e.g. checking whether two AFs win on the same campaigns) "
                          "doesn't depend on scrollback/terminal output.")
    args = ap.parse_args()

    oracle = DiscreteADACoatingsOracle.build()
    print(f"Oracle: {len(oracle)} real samples, {oracle.objective_names()} "
          f"({oracle.objective_directions()})")

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

    baseline = np.array(results["mo_egbo_novelty"])
    print(f"\n{'='*60}")
    if args.n_campaigns < 20:
        print(f"NOTE: n={args.n_campaigns} — Wilcoxon has essentially no power this "
              f"small (max achievable two-sided p at n=3 is 0.25; a mixed win/loss "
              f"split routinely gives p=1.0 regardless of the true effect). The "
              f"numbers below are for a timing/smoke check only — do not interpret "
              f"them, rerun with --n_campaigns 20 for anything conclusive.\n")
    print(f"Baseline (mo_egbo_novelty) mean final HV: {baseline.mean():.4g}")
    for cond_name in [c for c in CONDITIONS if c != "mo_egbo_novelty"]:
        af_hv = np.array(results[cond_name])
        diffs = af_hv - baseline
        pct = 100 * (af_hv.mean() - baseline.mean()) / baseline.mean()
        n_wins = int(np.sum(diffs > 0))
        try:
            _, pval = wilcoxon(diffs)
        except ValueError:
            pval = 1.0
        print(f"{cond_name:<16} mean_hv={af_hv.mean():.4g}  diff={pct:+.1f}%  "
              f"wins={n_wins}/{args.n_campaigns}  p={pval:.4f}")

    if args.n_campaigns < 20:
        print(f"\nThis was a {args.n_campaigns}-campaign PILOT — check the per-batch "
              f"timing above, then rerun with --n_campaigns 20 for the real comparison.")

    with open(args.out_path, "w") as f:
        json.dump({
            "n_campaigns": args.n_campaigns, "budget": args.budget,
            "n_init": args.n_init, "batch_size": args.batch_size, "seed": args.seed,
            "objective_names": oracle.objective_names(),
            "objective_directions": oracle.objective_directions(),
            "per_campaign_final_hv": results,  # {condition: [final_hv per campaign]}
        }, f, indent=2)
    print(f"\nSaved per-campaign results to {args.out_path}")


if __name__ == "__main__":
    main()
