"""
validate_l1.py — L1 build, step 5: held-out validation gate for the best
evolved AF from evolve_af.py.

Two criteria (per the confirmed L1 design):
  1. Beats EGBO-novelty on held-out campaigns by >=5 percentage points in
     mean per-campaign win-rate (i.e. mean win-rate >= 0.55), Wilcoxon
     signed-rank test of (per-campaign win-rate - 0.5) vs 0, p < 0.05.
  2. Per-campaign win-rate std <= 0.20 — i.e. it's not winning on average by
     dominating some campaigns and losing others.

Outcome matrix (as decided):
  Both pass            -> stable general MO AF found; L2 (online adaptation)
                           would add complexity without a clear target signal
                           — paper contribution is L1 + constraint-discovery
                           analysis, not L2.
  (1) passes, (2) fails -> the most interesting outcome: high per-campaign
                           variance despite winning on average IS the
                           campaign-specific signal, which is the strongest
                           possible justification for building L2 next.
  (1) fails              -> the AF-evolution idea is dead at this level.
                           Fall back to approach K (full code generation) or
                           approach I (DA-COREG) as the paper's main
                           contribution; report this as a negative result.

Usage:
    python validate_l1.py --af_path evolution_runs/run1/best_af.py \\
        --heldout_dir training_logs/heldout
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from evolve_af import load_training_steps
from af_interface import select_batch
from fitness_common import true_hv_gain_of_pick
from sandbox import run_af_in_sandbox, SandboxError


def per_campaign_win_rates(code: str, heldout_dir: pathlib.Path) -> dict:
    steps = load_training_steps(heldout_dir)
    by_campaign = {}
    for s in steps:
        by_campaign.setdefault(s["campaign"], []).append(s)

    rates = {}
    for campaign, camp_steps in by_campaign.items():
        wins, n_failed = 0, 0
        for s in camp_steps:
            try:
                scores = run_af_in_sandbox(
                    code, s["pool_x"], s["pool_mu"], s["pool_sigma"], s["X_obs"],
                    s["front_allmax"], s["ref_point_allmax"],
                    s["step"], s["budget"], s["n_obs"], s["stagnant_batches"],
                    s["Y_obs"],
                )
                af_idx = select_batch(scores, s["batch_size"])
                af_true_gain = true_hv_gain_of_pick(
                    af_idx, s["pool_x"], s["oracle_X"], s["oracle_Y"],
                    s["front_allmax"], s["ref_point_allmax"])
                if af_true_gain > s["egbo_true_gain"]:
                    wins += 1
            except SandboxError:
                n_failed += 1
        rates[campaign] = wins / len(camp_steps) if camp_steps else 0.0
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--af_path", default=str(pathlib.Path(__file__).parent /
                                              "evolution_runs" / "run1" / "best_af.py"))
    ap.add_argument("--heldout_dir", default=str(pathlib.Path(__file__).parent /
                                                  "training_logs" / "heldout"))
    args = ap.parse_args()

    code = pathlib.Path(args.af_path).read_text()
    print(f"Evaluating {args.af_path} on held-out campaigns in {args.heldout_dir} ...")

    rates_dict = per_campaign_win_rates(code, pathlib.Path(args.heldout_dir))
    campaigns = sorted(rates_dict)
    rates = np.array([rates_dict[c] for c in campaigns])

    if len(rates) == 0:
        print("No held-out campaigns found — run generate_training_set.py first.")
        return

    mean_rate, std_rate = rates.mean(), rates.std()
    print(f"\nHeld-out campaigns: {len(rates)}")
    print(f"Per-campaign win-rate: mean={mean_rate:.3f}  std={std_rate:.3f}  "
          f"min={rates.min():.3f}  max={rates.max():.3f}")

    diffs = rates - 0.5
    if np.all(diffs == 0):
        p_value = 1.0
    else:
        try:
            _, p_value = wilcoxon(diffs)
        except ValueError:
            p_value = 1.0  # all-zero or degenerate diffs

    crit1 = (mean_rate >= 0.55) and (p_value < 0.05)
    crit2 = std_rate <= 0.20

    print(f"\nCriterion 1 (mean win-rate >= 0.55, Wilcoxon p<0.05 vs 0.5): "
          f"mean={mean_rate:.3f}  p={p_value:.4f}  -> {'PASS' if crit1 else 'FAIL'}")
    print(f"Criterion 2 (per-campaign std <= 0.20): "
          f"std={std_rate:.3f}  -> {'PASS' if crit2 else 'FAIL'}")

    print()
    if crit1 and crit2:
        print("=> BOTH PASS: stable general MO AF found. L2's online adaptation "
              "would add complexity without a clear campaign-specific signal to "
              "exploit — paper contribution is L1 + constraint-discovery analysis.")
    elif crit1 and not crit2:
        print("=> CRITERION 1 PASSES, CRITERION 2 FAILS: the most interesting "
              "outcome. High per-campaign variance despite winning on average IS "
              "the campaign-specific signal — this is the strongest possible "
              "justification for building L2 (online adaptation) next.")
    else:
        print("=> CRITERION 1 FAILS: the AF-evolution idea is dead at this level "
              "(can't beat EGBO-novelty even on average, or not significantly). "
              "Fall back to approach K (full code generation) or approach I "
              "(DA-COREG) as the paper's main contribution; report L as a "
              "negative result.")


if __name__ == "__main__":
    main()
