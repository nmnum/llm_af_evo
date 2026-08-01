"""
diagnose_mu_sigma_dominance.py — one-shot diagnostic for why every
sigma-weighted AF variant collapsed to the same held-out selection on
coatings while trust_only (beta=0) alone differed. Not a re-run of
evolution — reads already-logged training steps and reports, per step,
whether pool_sigma has near-zero cross-candidate spread (hypothesis 1:
score = mu_sum + w*near-constant, so argsort(mu_sum + w*sigma_sum) ==
argsort(mu_sum) for ANY w > 0, which would also predict trust_only should
match — if it doesn't, this hypothesis is wrong) vs. sigma_sum being real
but rank-correlated with mu_sum strongly enough that the top-k boundary
gap in pure mu_sum dwarfs sigma_sum's range (hypothesis 2: a threshold
effect — mu dominates for any w in the range the hint set's weight
schedules actually produce, but a large enough w would eventually flip
it).

Deliberately does NOT reuse evolve_af.load_training_steps: that loader
hardcodes fitness_common.OBJECTIVE_DIRECTIONS (the 3-objective
Tm/kD/viscosity excipient convention) inside to_allmax, so it silently
mis-flips (or index-crashes, as it did here) on any other objective
count/oracle. Coatings training logs use the identical per-step schema
(pool_x_norm/pool_pred_mu/pool_pred_sigma etc., see
run_coatings_generalization.py/full_replay.py) but need the oracle's OWN
objective_directions() — obtained here via full_replay.reconstruct_oracle,
the same mechanism full_replay.py itself uses for coatings replay — rather
than a hardcoded default.

Usage: python diagnose_mu_sigma_dominance.py [train_dir] [--oracle-family coatings] [--n-steps N]
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

from fitness_common import to_allmax
from full_replay import reconstruct_oracle


def load_steps(train_dir: pathlib.Path, oracle_family: str) -> list:
    """
    Oracle-agnostic version of evolve_af.load_training_steps: flattens
    every campaign log's per-batch decisions into step records carrying
    pool_mu/pool_sigma/batch_size, using each log's OWN oracle (via
    reconstruct_oracle) for objective_directions rather than a hardcoded
    excipient default.
    """
    steps = []
    for f in sorted(train_dir.glob("*.json")):
        with open(f) as fh:
            log = json.load(fh)

        oracle = reconstruct_oracle(log, oracle_family=oracle_family)
        directions = oracle.objective_directions()

        Y_running = np.array(log["Y_init"])
        batch_size = log["batch_size"]

        for step in log["decisions"]:
            if "pool_x_norm" not in step or "pool_pred_sigma" not in step:
                if "picked_y" in step:
                    Y_running = np.vstack([Y_running, np.array(step["picked_y"])])
                continue

            steps.append({
                "campaign": f.name, "step": step["step"], "batch_size": batch_size,
                "pool_mu": to_allmax(np.array(step["pool_pred_mu"]), directions=directions),
                "pool_sigma": np.array(step["pool_pred_sigma"]),
            })

            if "picked_y" in step:
                Y_running = np.vstack([Y_running, np.array(step["picked_y"])])

    return steps


def diagnose_step(s: dict) -> dict:
    mu_sum = s["pool_mu"].sum(axis=1)
    sigma_sum = s["pool_sigma"].sum(axis=1)

    cv_sigma = float(sigma_sum.std() / (abs(sigma_sum.mean()) + 1e-12))
    rho, _ = spearmanr(mu_sum, sigma_sum)

    mu_order = np.argsort(-mu_sum)
    k = min(s["batch_size"], len(mu_sum) - 1)
    boundary_gap = float(mu_sum[mu_order[k - 1]] - mu_sum[mu_order[k]])
    sigma_range = float(sigma_sum.max() - sigma_sum.min())

    # Smallest uniform weight w such that w * sigma_sum alone could plausibly
    # overturn the mu-only top-k boundary (a lower bound, not exact — actual
    # flip also depends on WHICH candidates straddle the boundary).
    w_flip_lower_bound = boundary_gap / sigma_range if sigma_range > 1e-12 else float("inf")

    return {
        "n_pool": len(mu_sum),
        "mu_sum_range": float(mu_sum.max() - mu_sum.min()),
        "sigma_sum_range": sigma_range,
        "sigma_sum_cv": cv_sigma,
        "spearman_mu_sigma": float(rho) if rho is not None else float("nan"),
        "topk_boundary_gap_mu_only": boundary_gap,
        "w_flip_lower_bound": w_flip_lower_bound,
    }


def topk_set_changed(s: dict, w: float) -> bool:
    """
    Exact check (not a bound): does argsort(mu_sum + w*sigma_sum)'s top-k
    SET differ from argsort(mu_sum)'s top-k set, for this step's actual
    batch_size? This is what actually determines whether a weighted-AF's
    picks diverge from trust_only's — the boundary-gap/sigma-range bound
    is only a lower bound on w, not a statement about realistic weights.
    """
    mu_sum = s["pool_mu"].sum(axis=1)
    sigma_sum = s["pool_sigma"].sum(axis=1)
    k = min(s["batch_size"], len(mu_sum) - 1)

    mu_top = set(np.argsort(-mu_sum)[:k].tolist())
    weighted_top = set(np.argsort(-(mu_sum + w * sigma_sum))[:k].tolist())
    return mu_top != weighted_top


def sweep_weights(steps: list, idx: list, weights: list) -> dict:
    """
    Fraction of sampled steps where the top-k set at weight w differs from
    trust_only's (w=0) top-k set, for each w in weights.
    """
    return {
        w: float(np.mean([topk_set_changed(steps[i], w) for i in idx]))
        for w in weights
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", nargs="?", default="training_logs_coatings_100/train")
    ap.add_argument("--oracle-family", default="coatings", choices=["coatings", "excipient"])
    ap.add_argument("--n-steps", type=int, default=25,
                     help="Number of steps to sample (evenly spaced) from the training set.")
    ap.add_argument("--sweep-weights", type=float, nargs="+",
                     default=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
                     help="Weights to test for actual top-k SET changes vs. trust_only (w=0), "
                          "matching the magnitude range your hints' weight schedules produce.")
    args = ap.parse_args()

    train_dir = pathlib.Path(args.train_dir)
    if not train_dir.exists():
        print(f"train dir not found: {train_dir}", file=sys.stderr)
        sys.exit(1)

    steps = load_steps(train_dir, args.oracle_family)
    if not steps:
        print("no usable steps found (need pool_x_norm/pool_pred_sigma in logs)", file=sys.stderr)
        sys.exit(1)

    idx = list(range(0, len(steps), max(1, len(steps) // args.n_steps)))[: args.n_steps]

    rows = [diagnose_step(steps[i]) for i in idx]

    def summarize(key):
        vals = np.array([r[key] for r in rows])
        return vals.mean(), np.median(vals), vals.min(), vals.max()

    print(f"{len(rows)} steps sampled from {len(steps)} total, train_dir={train_dir}, "
          f"oracle_family={args.oracle_family}\n")
    print(f"{'metric':30s} {'mean':>10s} {'median':>10s} {'min':>10s} {'max':>10s}")
    for key in ["n_pool", "mu_sum_range", "sigma_sum_range", "sigma_sum_cv",
                "spearman_mu_sigma", "topk_boundary_gap_mu_only", "w_flip_lower_bound"]:
        m, med, lo, hi = summarize(key)
        print(f"{key:30s} {m:10.4g} {med:10.4g} {lo:10.4g} {hi:10.4g}")

    print(f"\nExact top-k SET change vs. trust_only (w=0), fraction of "
          f"{len(idx)} sampled steps where it differs:")
    print(f"{'weight w':>10s} {'frac steps changed':>20s}")
    frac_changed = sweep_weights(steps, idx, args.sweep_weights)
    for w, frac in frac_changed.items():
        print(f"{w:10.3g} {frac:20.3f}")

    print("""
Reading the output:
  sigma_sum_cv near 0            -> hypothesis 1 (near-constant sigma across
                                     the pool): argsort(mu_sum + w*sigma_sum)
                                     == argsort(mu_sum) for EVERY w > 0. This
                                     predicts trust_only (w=0) should ALSO
                                     match every nonzero-weight AF -- if it
                                     doesn't in your held-out results, this
                                     hypothesis is wrong regardless of what
                                     the CV number says.
  spearman_mu_sigma strongly     -> sigma is rank-correlated with mu (not
  positive or negative              independent noise) -- consistent with
                                     the pool being generated by an
                                     acquisition-aware optimizer that already
                                     trades the two off.
  w_flip_lower_bound >> the      -> hypothesis 2 (threshold effect): mu
  weight magnitudes your 8          dominates the top-k ranking for any w
  hints' schedules actually         your hint set's weight schedules
  produce                            actually produce, and would need a much
                                     larger w to ever flip a single top-k
                                     boundary member. This is a LOWER BOUND
                                     (assumes only the single boundary pair
                                     matters), so a real flip may need even
                                     more than this to move the FULL batch.

Weight-sweep table above (the exact test, not a bound): if frac_steps_changed
stays near 0 even at w values matching or exceeding your hints' actual
weight schedules (fixed_ucb's beta=2.0, etc.), that confirms the sharper
version of hypothesis 2 directly: the specific candidates competing at the
top-k boundary have near-identical sigma even though sigma varies a lot
pool-wide, so realistic weights don't flip the SET despite sigma's real
overall spread. If frac_steps_changed rises sharply somewhere in this
range instead, sigma DOES matter at realistic weights, and byte-identical
held-out results would then point at something else (e.g. every hint's
generated code converging to a similar effective weight schedule, or a
bug in how the weight is applied) rather than mu-sum dominance.
""")


if __name__ == "__main__":
    main()
