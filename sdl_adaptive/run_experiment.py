"""
run_experiment.py — Run the full SDL adaptive strategy experiment.

Usage
-----
# Mock LLM controllers only (no Ollama required):
python run_experiment.py --n_seeds 20 --out_dir results/

# Include real LLM controllers (requires Ollama running with qwen2.5-coder:7b):
python run_experiment.py --n_seeds 20 --out_dir results/ --llm

# Single dataset, quick test:
python run_experiment.py --n_seeds 3 --datasets coatings --out_dir results_test/

# Merge real LLM results into existing mock results:
python run_experiment.py --n_seeds 20 --out_dir results/ --llm --llm_only

Output
------
results/
  metrics_summary.csv          — 820 rows (9 conditions × 5 datasets × 20 seeds)
  <dataset>/<condition>/
    curves.npy                 — (n_seeds, budget) running-best curves
    metrics_agg.json           — aggregated metrics (mean ± std)
    metrics_per_seed.json      — per-seed metrics list
    switch_logs.json           — per-seed decision logs
"""

import argparse
import json
import logging
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from oracle import NNOracle
from simulator import CampaignSimulator
from baselines import FixedStrategyBaseline, ADAOriginalBaseline
from evaluate import compute_metrics, aggregate_across_seeds
from controllers.mock_controller import (
    MockApproachAController, MockApproachBController, MockApproachCController,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset registry ─────────────────────────────────────────────────────────
ALL_DATASETS = [
    ("coatings",        "coatings"),
    ("pareto_20201218", "pareto_campaign 2020-12-18_17-38-40"),
    ("pareto_20201223", "pareto_campaign 2020-12-23_17-06-50"),
    ("pareto_20210104", "pareto_campaign 2021-01-04_08-37-39"),
    ("pareto_20210112", "pareto_campaign 2021-01-12_16-26-56"),
]

# ── Controller factory ────────────────────────────────────────────────────────
def make_controller(condition: str, seed: int, ds_name: str,
                    ada_csv: str = None, model: str = "qwen2.5-coder:7b"):
    if condition == "fixed_random":    return FixedStrategyBaseline("random", {})
    if condition == "fixed_ucb_low":   return FixedStrategyBaseline("ucb", {"beta": 0.2})
    if condition == "fixed_ucb_high":  return FixedStrategyBaseline("ucb", {"beta": 400.0})
    if condition == "fixed_ei":        return FixedStrategyBaseline("ei", {})
    if condition == "fixed_lhs":       return FixedStrategyBaseline("lhs", {})
    if condition == "ada_original" and ada_csv:
        return ADAOriginalBaseline.from_coatings_csv(ada_csv)
    if condition == "mock_approach_a": return MockApproachAController(seed=seed)
    if condition == "mock_approach_b": return MockApproachBController(seed=seed)
    if condition == "mock_approach_c": return MockApproachCController(seed=seed)

    # Real LLM controllers (require Ollama)
    if condition == "approach_a":
        from controllers.approach_a import ApproachAController
        return ApproachAController(model=model)
    if condition == "approach_b":
        from controllers.approach_b import ApproachBController
        return ApproachBController(model=model)
    if condition == "approach_c":
        from controllers.approach_c import ApproachCController
        return ApproachCController(model=model)

    raise ValueError(f"Unknown condition: {condition!r}")


# ── Main runner ───────────────────────────────────────────────────────────────
def run_experiment(
    data_dir: str,
    out_dir: str,
    n_seeds: int = 20,
    datasets: list = None,
    conditions: list = None,
    model: str = "qwen2.5-coder:7b",
    discrete: bool = False,
):
    data_dir = pathlib.Path(data_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_list = ALL_DATASETS
    if datasets:
        dataset_list = [(l, n) for l, n in ALL_DATASETS if l in datasets]

    mock_conditions = [
        "fixed_random", "fixed_ucb_low", "fixed_ucb_high",
        "fixed_ei", "fixed_lhs",
        "mock_approach_a", "mock_approach_b", "mock_approach_c",
    ]
    if conditions is None:
        conditions = mock_conditions

    all_rows = []
    t0 = time.time()

    for ds_label, ds_name in dataset_list:
        print(f"\n{'='*55}\nDataset: {ds_label}")
        oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        gb = oracle.global_best()
        gmin = float(oracle._y_raw.min())
        budget = len(oracle._X_raw)
        ada_csv = str(data_dir / "coatings_2022.csv") if ds_name == "coatings" else None
        print(f"  n={len(oracle._X_raw)}, dims={oracle.bounds().shape[0]}, "
              f"global_best={gb:.3f}, budget={budget}")

        ds_conditions = conditions[:]
        if ds_name == "coatings" and "ada_original" not in ds_conditions:
            ds_conditions.append("ada_original")

        for condition in ds_conditions:
            seed_metrics, seed_curves, seed_logs = [], [], []

            for seed in range(n_seeds):
                try:
                    ctrl = make_controller(condition, seed, ds_name, ada_csv, model)
                except Exception as e:
                    logger.warning(f"Skipping {condition} seed {seed}: {e}")
                    continue

                sim = CampaignSimulator(
                    oracle, ds_name, n_init=5, budget=budget,
                    controller_interval=5, seed=seed,
                    discrete=discrete,
                )
                res = sim.run(ctrl)
                m = compute_metrics(res, gb, budget=budget, oracle_global_min=gmin)
                seed_metrics.append(m)
                seed_curves.append(np.array(res["running_best"]))

                # Serialise decisions (strip numpy arrays)
                decisions_clean = []
                for t, s, p in res["decisions"]:
                    p2 = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in p.items()}
                    decisions_clean.append((t, s, p2))
                seed_logs.append({"decisions": decisions_clean, "failures": res["failures"]})

            if not seed_metrics:
                continue

            agg = aggregate_across_seeds(seed_metrics)
            cond_dir = out_dir / ds_label / condition
            cond_dir.mkdir(parents=True, exist_ok=True)

            np.save(cond_dir / "curves.npy", np.array(seed_curves))
            with open(cond_dir / "metrics_agg.json", "w") as f:
                json.dump(agg, f, indent=2)
            with open(cond_dir / "metrics_per_seed.json", "w") as f:
                json.dump(seed_metrics, f, indent=2)
            with open(cond_dir / "switch_logs.json", "w") as f:
                json.dump(seed_logs, f, indent=2)

            print(f"  {condition:22s}: auc={agg['auc_best']['mean']:.3f}"
                  f"±{agg['auc_best']['std']:.3f}"
                  f"  final={agg['final_best_normalised']['mean']:.3f}"
                  f"  sw={agg['switch_count']['mean']:.1f}"
                  f"  fail={agg['failure_count']['mean']:.1f}")

            for i, m in enumerate(seed_metrics):
                row = {"dataset": ds_label, "condition": condition, "seed": i,
                       "global_best": gb, "global_min": gmin}
                row.update(m)
                all_rows.append(row)

    # Save / merge summary CSV
    summary_path = out_dir / "metrics_summary.csv"
    new_df = pd.DataFrame(all_rows)

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Remove rows for conditions we just ran (overwrite)
        mask = existing.apply(
            lambda r: (r["dataset"], r["condition"]) in
            {(row["dataset"], row["condition"]) for row in all_rows},
            axis=1,
        )
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(summary_path, index=False)
        print(f"\nMerged into {summary_path} ({len(combined)} rows total)")
    else:
        new_df.to_csv(summary_path, index=False)
        print(f"\nSaved {summary_path} ({len(new_df)} rows)")

    print(f"Total runtime: {time.time() - t0:.0f}s")
    return new_df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDL adaptive strategy experiment")
    parser.add_argument("--data_dir", default="data", help="Path to ADA data directory")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    parser.add_argument("--n_seeds", type=int, default=20, help="Number of random seeds")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of datasets (default: all 5)")
    parser.add_argument("--llm", action="store_true",
                        help="Include real LLM conditions (requires Ollama)")
    parser.add_argument("--llm_only", action="store_true",
                        help="Run only real LLM conditions (merge into existing results)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Explicit list of conditions to run, e.g. --conditions fixed_ucb_low fixed_lhs fixed_ei")
    parser.add_argument("--model", default="qwen2.5-coder:7b",
                        help="Ollama model name")
    parser.add_argument("--discrete", action="store_true",
                        help="Use discrete scoring (score unqueried dataset rows directly)")
    args = parser.parse_args()

    conditions = None  # default: all mock conditions
    if args.conditions:
        conditions = args.conditions
    elif args.llm_only:
        conditions = ["approach_c"] #["approach_a", "approach_b", "approach_c"]
    elif args.llm:
        conditions = [
            "fixed_random", "fixed_ucb_low", "fixed_ucb_high",
            "fixed_ei", "fixed_lhs",
            "mock_approach_a", "mock_approach_b", "mock_approach_c",
            "approach_a", "approach_b", "approach_c",
        ]

    run_experiment(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_seeds=args.n_seeds,
        datasets=args.datasets,
        conditions=conditions,
        model=args.model,
        discrete=args.discrete,
    )
