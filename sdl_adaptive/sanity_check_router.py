"""
sanity_check_router.py — 3-seed validation of approach_d before full Phase 1 run.

Runs approach_d and rule_router on pareto_20210112 for 3 seeds.
Prints the actual strategy sequences to verify the LLM is routing sensibly
rather than anchoring to one strategy.

Usage:
    python sanity_check_router.py --data_dir data/ --model qwen2.5-coder:7b-instruct

Pass criteria (all must hold before running Phase 1):
  1. approach_d produces at least 2 different strategies across the 3 seeds
  2. approach_d switches strategy at least once per seed on average
  3. No sandbox errors or LLM timeouts
  4. rule_router shows sensible switching (not stuck on lhs or ucb throughout)
"""

import argparse
import pathlib
import sys
import json

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from oracle import NNOracle
from shared_seed_experiment import (
    generate_shared_inits, run_sdl_campaign, make_controller
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="pareto_20210112")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--budget_frac", type=float, default=0.5)
    parser.add_argument("--n_init", type=int, default=5)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    ds_name_map = {
        "pareto_20210112": "pareto_campaign 2021-01-12_16-26-56",
        "pareto_20201218": "pareto_campaign 2020-12-18_17-38-40",
        "coatings":        "coatings",
    }
    ds_name = ds_name_map.get(args.dataset, args.dataset)

    oracle = NNOracle.from_dataset(ds_name, str(data_dir))
    N = len(oracle._X_raw)
    budget = max(args.n_init + 10, int(args.budget_frac * N))
    gb = oracle.global_best()

    print(f"Dataset: {args.dataset}  N={N}  budget={budget}  "
          f"dims={oracle.bounds().shape[0]}  global_best={gb:.4f}")

    shared_inits = generate_shared_inits(oracle, args.n_seeds, args.n_init, rng_seed=42)
    best_norms = [y.max()/gb for _, y in shared_inits]
    print(f"Init best_norm: {[f'{b:.2f}' for b in best_norms]}")
    print()

    for condition in ["rule_router", "approach_d"]:
        print(f"{'='*55}")
        print(f"Condition: {condition}")

        all_strategies = []
        issues = []

        for seed_idx, (X_init, y_init) in enumerate(shared_inits):
            try:
                ctrl = make_controller(condition, args.model)
            except Exception as e:
                print(f"  Seed {seed_idx}: FAILED to create controller: {e}")
                continue

            res = run_sdl_campaign(
                oracle, ds_name, X_init, y_init, ctrl,
                budget, controller_interval=5
            )

            strategies = [s for _, s, _ in res["decisions"]]
            params_list = [p for _, _, p in res["decisions"]]
            steps = [t for t, _, _ in res["decisions"]]
            switches = sum(strategies[i] != strategies[i-1]
                          for i in range(1, len(strategies)))
            all_strategies.extend(strategies)

            if res["failures"]:
                issues.append(f"seed {seed_idx}: {len(res['failures'])} failures")

            # Print decision sequence
            seq_parts = []
            for t, s, p in zip(steps, strategies, params_list):
                if s == "ucb" and "beta" in p:
                    seq_parts.append(f"t{t}:{s}(β={p['beta']:.1f})")
                else:
                    seq_parts.append(f"t{t}:{s}")
            print(f"  Seed {seed_idx} (switches={switches}): "
                  f"{' → '.join(seq_parts)}")

            # Print lengthscale values if available (via decisions context)
            auc = res["running_best"][-1] / gb
            print(f"    final_best_norm={auc:.3f}  failures={res['failures']}")

        # Summary
        from collections import Counter
        counts = Counter(all_strategies)
        unique = len(counts)
        print(f"\n  Strategy distribution: {dict(counts)}")
        print(f"  Unique strategies used: {unique}")
        if issues:
            print(f"  Issues: {issues}")

        # Pass/fail
        print()
        if condition == "approach_d":
            if unique >= 2:
                print(f"  ✓ PASS: LLM used {unique} different strategies")
            else:
                print(f"  ✗ FAIL: LLM anchored to one strategy — "
                      f"consider exploration-score fallback")
            avg_switches = sum(
                sum(strategies[i] != strategies[i-1]
                    for i in range(1, len(strategies)))
                for strategies in [
                    [s for _, s, _ in run_sdl_campaign(
                        oracle, ds_name, X_init, y_init,
                        make_controller(condition, args.model),
                        budget, controller_interval=5
                    )["decisions"]]
                    for X_init, y_init in shared_inits[:1]
                ]
            )
        print()

    print("If approach_d PASS: run Phase 1 with full n_repeats=20")
    print("If approach_d FAIL: apply exploration-score fallback in approach_d.py")


if __name__ == "__main__":
    main()
