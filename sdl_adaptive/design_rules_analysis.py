"""
design_rules_analysis.py — Extract design rules from simulation results
before running real experiments.

Analyses which β schedules (approach_a) and score patterns (approach_b)
were associated with fast convergence vs slow convergence across seeds.
Run this after the 20-seed simulation to calibrate prompts for real experiments.

Usage:
    python design_rules_analysis.py \
        --results_dir results/ \
        --dataset coatings \
        --out design_rules.json
"""

import argparse
import json
import pathlib
import numpy as np


def analyse_approach_a(results_dir: pathlib.Path, dataset: str):
    """
    For approach_a: compare β schedules of fast vs slow seeds.
    Fast = steps_to_90pct in bottom quartile (found optimum quickly).
    Slow = steps_to_90pct in top quartile or never reached 90%.
    """
    logs_path = results_dir / dataset / "approach_a" / "switch_logs.json"
    metrics_path = results_dir / dataset / "approach_a" / "metrics_per_seed.json"

    if not logs_path.exists() or not metrics_path.exists():
        print(f"  Missing files for approach_a / {dataset}")
        return {}

    with open(logs_path) as f:
        logs = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    steps_to_90 = [m["steps_to_90pct"] for m in metrics]
    aucs = [m["auc_best"] for m in metrics]

    # Separate fast and slow seeds
    # Fast: reached 90% quickly (bottom third of those that reached it)
    reached = [(i, s) for i, s in enumerate(steps_to_90) if s >= 0]
    never = [i for i, s in enumerate(steps_to_90) if s < 0]

    if len(reached) >= 4:
        reached_sorted = sorted(reached, key=lambda x: x[1])
        n_fast = max(1, len(reached) // 3)
        fast_seeds = [i for i, _ in reached_sorted[:n_fast]]
        slow_seeds = [i for i, _ in reached_sorted[-n_fast:]] + never
    else:
        # Not enough seeds reached 90% — split by AUC quartile
        auc_sorted = sorted(enumerate(aucs), key=lambda x: x[1])
        n = len(auc_sorted)
        fast_seeds = [i for i, _ in auc_sorted[n*3//4:]]
        slow_seeds = [i for i, _ in auc_sorted[:n//4]]

    def extract_beta_schedule(seed_idx):
        if seed_idx >= len(logs):
            return []
        decisions = logs[seed_idx]["decisions"]
        return [(t, p.get("beta", None)) for t, s, p in decisions
                if p.get("beta") is not None]

    print(f"\n  approach_a on {dataset}:")
    print(f"  Fast seeds ({len(fast_seeds)}): {fast_seeds}")
    print(f"  Slow/failed seeds ({len(slow_seeds)}): {slow_seeds}")

    # Compute mean β at each step for fast vs slow
    fast_betas = [extract_beta_schedule(i) for i in fast_seeds]
    slow_betas = [extract_beta_schedule(i) for i in slow_seeds]

    # Find steps common to all seeds
    if fast_betas and fast_betas[0]:
        steps = [t for t, _ in fast_betas[0]]
        budget = max(steps) + 1 if steps else 91

        print(f"\n  β schedule comparison (fast vs slow seeds):")
        print(f"  {'Step':>6}  {'Progress':>10}  {'Fast β (mean)':>15}  {'Slow β (mean)':>15}  {'Signal':>10}")

        rules = []
        for step in steps:
            progress = step / budget
            fast_b = [dict(sched).get(step) for sched in fast_betas
                      if dict(sched).get(step) is not None]
            slow_b = [dict(sched).get(step) for sched in slow_betas
                      if dict(sched).get(step) is not None]

            if fast_b and slow_b:
                fb = np.mean(fast_b)
                sb = np.mean(slow_b)
                ratio = fb / sb if sb > 0 else float('inf')
                signal = "explore more" if ratio > 1.3 else (
                         "exploit more" if ratio < 0.77 else "similar")
                print(f"  {step:>6}  {100*progress:>9.0f}%  {fb:>15.1f}  {sb:>15.1f}  {signal:>10}")
                rules.append({
                    "step": step, "progress": round(progress, 2),
                    "fast_beta_mean": round(fb, 2), "slow_beta_mean": round(sb, 2),
                    "recommendation": signal
                })

        return {"fast_seeds": fast_seeds, "slow_seeds": slow_seeds, "beta_rules": rules}

    return {}


def analyse_approach_b(results_dir: pathlib.Path, dataset: str):
    """
    For approach_b: which score values were associated with fast convergence?
    """
    logs_path = results_dir / dataset / "approach_b" / "switch_logs.json"
    metrics_path = results_dir / dataset / "approach_b" / "metrics_per_seed.json"

    if not logs_path.exists() or not metrics_path.exists():
        return {}

    with open(logs_path) as f:
        logs = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)

    aucs = [m["auc_best"] for m in metrics]
    steps_to_90 = [m["steps_to_90pct"] for m in metrics]

    # Strategy sequence analysis: what does a high-AUC seed look like?
    top_seeds = sorted(range(len(aucs)), key=lambda i: aucs[i], reverse=True)[:5]
    bot_seeds = sorted(range(len(aucs)), key=lambda i: aucs[i])[:5]

    print(f"\n  approach_b on {dataset}:")
    print(f"  Top-5 AUC seeds: {top_seeds} (aucs: {[round(aucs[i],3) for i in top_seeds]})")
    print(f"  Bot-5 AUC seeds: {bot_seeds} (aucs: {[round(aucs[i],3) for i in bot_seeds]})")

    def get_strategy_seq(seed_idx):
        if seed_idx >= len(logs):
            return []
        return [(t, s) for t, s, p in logs[seed_idx]["decisions"]]

    print(f"\n  Strategy sequences (top-5 seeds):")
    for i in top_seeds[:3]:
        seq = " → ".join(f"t{t}:{s}" for t, s in get_strategy_seq(i))
        print(f"    seed {i} (auc={aucs[i]:.3f}): {seq}")

    print(f"\n  Strategy sequences (bot-5 seeds):")
    for i in bot_seeds[:3]:
        seq = " → ".join(f"t{t}:{s}" for t, s in get_strategy_seq(i))
        print(f"    seed {i} (auc={aucs[i]:.3f}): {seq}")

    return {"top_seeds": top_seeds, "bot_seeds": bot_seeds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--dataset", default="coatings")
    parser.add_argument("--out", default="design_rules.json")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)

    print(f"Design rules analysis for: {args.dataset}")
    print("=" * 60)

    all_rules = {}

    rules_a = analyse_approach_a(results_dir, args.dataset)
    if rules_a:
        all_rules["approach_a"] = rules_a

    rules_b = analyse_approach_b(results_dir, args.dataset)
    if rules_b:
        all_rules["approach_b"] = rules_b

    # Summary recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION FOR REAL EXPERIMENT PROMPT CALIBRATION")
    print(f"{'='*60}")

    if "approach_a" in all_rules and all_rules["approach_a"].get("beta_rules"):
        br = all_rules["approach_a"]["beta_rules"]
        early = [r for r in br if r["progress"] < 0.3]
        late  = [r for r in br if r["progress"] > 0.7]
        if early:
            eb = np.mean([r["fast_beta_mean"] for r in early])
            print(f"\napproach_a: Fast seeds used β≈{eb:.0f} in early phase (progress<30%)")
        if late:
            lb = np.mean([r["fast_beta_mean"] for r in late])
            print(f"approach_a: Fast seeds used β≈{lb:.1f} in late phase (progress>70%)")
        print("→ Update system_a.txt ranges to match these empirical values")

    with open(args.out, "w") as f:
        json.dump(all_rules, f, indent=2)
    print(f"\nFull rules saved to {args.out}")


if __name__ == "__main__":
    main()
