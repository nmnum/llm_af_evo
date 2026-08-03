"""
check_confound.py — Check whether fast seeds had higher best_normalised
at the critical steps (10-20% progress) than slow seeds.

Uses curves.npy (running-best per seed) and metrics_per_seed.json
to reconstruct best_normalised at each step without needing extra logging.

Usage:
    python check_confound.py --results_dir results/ --dataset coatings
"""

import argparse
import json
import pathlib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dataset", default="coatings")
    parser.add_argument("--condition", default="approach_a")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    cond_dir = results_dir / args.dataset / args.condition

    curves = np.load(cond_dir / "curves.npy")          # (n_seeds, budget)
    with open(cond_dir / "metrics_per_seed.json") as f:
        metrics = json.load(f)

    n_seeds, budget = curves.shape
    global_best = metrics[0]["final_best"] / metrics[0]["final_best_normalised"]

    # Reconstruct best_normalised at each step
    curves_norm = curves / global_best   # (n_seeds, budget)

    # Identify fast vs slow seeds (same logic as design_rules_analysis.py)
    steps_to_90 = [m["steps_to_90pct"] for m in metrics]
    aucs = [m["auc_best"] for m in metrics]

    reached = [(i, s) for i, s in enumerate(steps_to_90) if s >= 0]
    never   = [i for i, s in enumerate(steps_to_90) if s < 0]

    if len(reached) >= 4:
        reached_sorted = sorted(reached, key=lambda x: x[1])
        n_split = max(1, len(reached) // 3)
        fast_seeds = [i for i, _ in reached_sorted[:n_split]]
        slow_seeds = [i for i, _ in reached_sorted[-n_split:]] + never[:n_split]
    else:
        auc_sorted = sorted(enumerate(aucs), key=lambda x: x[1])
        n = len(auc_sorted)
        fast_seeds = [i for i, _ in auc_sorted[n*3//4:]]
        slow_seeds = [i for i, _ in auc_sorted[:n//4]]

    print(f"Dataset: {args.dataset}  |  Condition: {args.condition}")
    print(f"Budget: {budget}  |  Global best: {global_best:.4f}")
    print(f"Fast seeds: {fast_seeds}")
    print(f"Slow seeds: {slow_seeds}")
    print()

    # Controller calls happen every 5 steps starting from n_init=5
    controller_steps = list(range(5, budget, 5))

    print(f"{'Step':>6}  {'Progress':>9}  "
          f"{'Fast best_norm (mean)':>22}  {'Slow best_norm (mean)':>22}  "
          f"{'Δ (fast-slow)':>14}  {'Verdict':>20}")
    print("-" * 100)

    for step in controller_steps:
        if step >= budget:
            break
        progress = step / budget
        fast_bn = curves_norm[fast_seeds, step].mean()
        slow_bn = curves_norm[slow_seeds, step].mean()
        delta = fast_bn - slow_bn

        if delta > 0.05:
            verdict = "fast MUCH better ←"
        elif delta > 0.02:
            verdict = "fast slightly better"
        elif delta < -0.05:
            verdict = "slow MUCH better"
        elif delta < -0.02:
            verdict = "slow slightly better"
        else:
            verdict = "similar"

        print(f"  {step:>4}  {100*progress:>8.0f}%  "
              f"{fast_bn:>22.3f}  {slow_bn:>22.3f}  "
              f"{delta:>+14.3f}  {verdict:>20}")

    print()
    print("=== INTERPRETATION ===")
    print()
    print("If fast seeds have best_normalised >> slow seeds at steps 10-20:")
    print("  → LUCK CONFOUND: fast seeds got better initial points.")
    print("    The β difference in design_rules.json is the LLM reacting")
    print("    to a better campaign state, not causing better outcomes.")
    print("    Do NOT change the prompt based on design_rules.json.")
    print()
    print("If best_normalised is SIMILAR between fast and slow at steps 10-20:")
    print("  → GENUINE SIGNAL: the LLM's β choices are causing the difference.")
    print("    The earlier transition to exploitation (step 15-20) is causal.")
    print("    Consider tightening the prompt ranges based on design_rules.json.")


if __name__ == "__main__":
    main()
