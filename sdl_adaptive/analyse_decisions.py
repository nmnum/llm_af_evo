"""
analyse_decisions.py — Inspect what strategies the router actually selected.

Reads switch_logs.json for a condition across all datasets and answers:
  - What fraction of decisions were egbo vs ucb vs lhs vs random?
  - On structured datasets (hartmann3, hartmann6, pareto_20210112),
    did the router select egbo? How often?
  - On flat datasets (pareto_20201218), did it avoid egbo?
  - What beta values did the LLM choose for ucb?

Usage:
    python analyse_decisions.py --results_dir results_phase1/ --condition approach_d
    python analyse_decisions.py --results_dir results_phase1/ --condition rule_router
"""

import argparse
import json
import pathlib
import numpy as np
from collections import Counter, defaultdict

STRUCTURED = {"coatings", "pareto_20210112", "hartmann3", "hartmann6"}
FLAT       = {"pareto_20201218"}

LABELS = {
    "coatings": "coatings (7D, structured)",
    "pareto_20201218": "pareto_20201218 (4D, FLAT)",
    "pareto_20210112": "pareto_20210112 (4D, structured)",
    "hartmann3": "hartmann3 (3D, structured)",
    "hartmann6": "hartmann6 (6D, structured)",
}


def load_logs(results_dir: pathlib.Path, dataset: str, condition: str):
    p = results_dir / dataset / condition / "switch_logs.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def analyse_dataset(logs, dataset: str, condition: str):
    """Returns per-seed decision sequences and aggregate stats."""
    all_strategies = []
    all_betas = []
    egbo_seeds = 0   # seeds where egbo was selected at least once
    egbo_first = 0   # seeds where egbo was the FIRST decision

    for seed_idx, seed_log in enumerate(logs):
        decisions = seed_log.get("decisions", [])
        strategies = []
        betas = []
        for dec in decisions:
            if isinstance(dec, (list, tuple)) and len(dec) >= 2:
                step, strategy = dec[0], dec[1]
                params = dec[2] if len(dec) > 2 else {}
            elif isinstance(dec, dict):
                strategy = dec.get("strategy", "unknown")
                params = dec.get("params", {})
            else:
                continue

            # Strip any suffix like "egbo_llm:ucb(b=2.0)"
            base = str(strategy).split(":")[0].split("(")[0].lower()
            if base in ("egbo", "novelty_egbo"):
                base = "egbo"
            strategies.append(base)

            if base == "ucb" and isinstance(params, dict) and "beta" in params:
                betas.append(float(params["beta"]))

        all_strategies.extend(strategies)
        all_betas.extend(betas)

        if "egbo" in strategies:
            egbo_seeds += 1
        if strategies and strategies[0] == "egbo":
            egbo_first += 1

    n_seeds = len(logs)
    counts = Counter(all_strategies)
    total = sum(counts.values())

    return {
        "n_seeds": n_seeds,
        "total_decisions": total,
        "strategy_counts": dict(counts),
        "strategy_pct": {s: 100*c/total for s,c in counts.items()} if total else {},
        "egbo_seeds": egbo_seeds,
        "egbo_seeds_pct": 100*egbo_seeds/n_seeds if n_seeds else 0,
        "egbo_first": egbo_first,
        "mean_beta": float(np.mean(all_betas)) if all_betas else None,
        "beta_values": all_betas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_phase1")
    parser.add_argument("--condition",   default="approach_d")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    condition   = args.condition

    datasets = [d.name for d in results_dir.iterdir()
                if d.is_dir() and (d / condition / "switch_logs.json").exists()]
    datasets.sort()

    if not datasets:
        print(f"No switch_logs.json found for condition '{condition}' in {results_dir}")
        return

    print(f"Decision analysis: {condition}")
    print(f"Results dir: {results_dir}")
    print(f"Datasets found: {datasets}")
    print()

    # ── Per-dataset breakdown ────────────────────────────────────────────────
    structured_egbo_rates = []
    flat_egbo_rates       = []

    for ds in datasets:
        logs = load_logs(results_dir, ds, condition)
        if logs is None:
            continue

        stats = analyse_dataset(logs, ds, condition)
        label = LABELS.get(ds, ds)
        regime = "STRUCTURED" if ds in STRUCTURED else "FLAT" if ds in FLAT else "?"

        print(f"{'='*60}")
        print(f"{label}  [{regime}]  n={stats['n_seeds']} seeds")
        print(f"  Total decisions: {stats['total_decisions']}")
        print()

        # Strategy distribution
        print(f"  Strategy distribution:")
        for s, pct in sorted(stats["strategy_pct"].items(),
                              key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            print(f"    {s:<12} {pct:>5.1f}%  {bar}")

        print()
        print(f"  Seeds that used egbo at least once: "
              f"{stats['egbo_seeds']}/{stats['n_seeds']} "
              f"({stats['egbo_seeds_pct']:.0f}%)")
        print(f"  Seeds where egbo was FIRST decision: {stats['egbo_first']}")

        if stats["mean_beta"] is not None:
            betas = stats["beta_values"]
            print(f"  UCB beta — mean={stats['mean_beta']:.2f}  "
                  f"min={min(betas):.2f}  max={max(betas):.2f}  "
                  f"n={len(betas)}")
        print()

        if regime == "STRUCTURED":
            structured_egbo_rates.append(stats["egbo_seeds_pct"])
        elif regime == "FLAT":
            flat_egbo_rates.append(stats["egbo_seeds_pct"])

        # Verbose: show per-seed sequences
        if args.verbose:
            logs2 = load_logs(results_dir, ds, condition)
            print(f"  Per-seed sequences (first 5 seeds):")
            for i, seed_log in enumerate(logs2[:5]):
                decisions = seed_log.get("decisions", [])
                seq = []
                for dec in decisions:
                    if isinstance(dec, (list, tuple)) and len(dec) >= 2:
                        s = str(dec[1]).split(":")[0].split("(")[0][:4]
                        p = dec[2] if len(dec) > 2 else {}
                        if isinstance(p, dict) and "beta" in p:
                            seq.append(f"{s}(β{p['beta']:.0f})")
                        else:
                            seq.append(s)
                    elif isinstance(dec, dict):
                        s = str(dec.get("strategy","?"))[:4]
                        seq.append(s)
                print(f"    seed {i}: {' → '.join(seq)}")
            print()

    # ── Summary: routing quality assessment ──────────────────────────────────
    print(f"{'='*60}")
    print("ROUTING QUALITY ASSESSMENT")
    print(f"{'='*60}")
    print()

    if structured_egbo_rates:
        mean_struct = np.mean(structured_egbo_rates)
        print(f"  On STRUCTURED datasets: egbo selected in "
              f"{mean_struct:.0f}% of seeds (mean across datasets)")
        if mean_struct < 20:
            print(f"  ✗ FAILURE MODE A: router rarely selects egbo on structured landscapes")
            print(f"    → The routing signal (lengthscale_norm at n_init=5) is too noisy")
            print(f"    → Fix: increase n_init to 15-20 before first routing call")
        elif mean_struct > 60:
            print(f"  ✓ Router frequently selects egbo on structured landscapes")
            print(f"    → If AUC still underperforms always-EGBO, this is FAILURE MODE B")
            print(f"      (switching cost — budget wasted on wrong strategy before routing)")
        else:
            print(f"  ~ Partial routing: egbo selected sometimes but not consistently")

    if flat_egbo_rates:
        mean_flat = np.mean(flat_egbo_rates)
        print()
        print(f"  On FLAT datasets: egbo selected in "
              f"{mean_flat:.0f}% of seeds (should be LOW)")
        if mean_flat > 40:
            print(f"  ✗ Router over-uses egbo on flat landscapes → routing penalty")
        else:
            print(f"  ✓ Router correctly avoids egbo on flat landscapes")

    print()
    print("  Interpretation guide:")
    print("  Failure mode A (wrong decisions): fix with better signal quality (n_init↑)")
    print("  Failure mode B (switching cost):  fix with prior-knowledge routing")
    print("  Both modes present: need architectural change, not prompt tuning")


if __name__ == "__main__":
    main()
