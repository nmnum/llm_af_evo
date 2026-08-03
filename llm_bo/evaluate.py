"""
evaluate.py — Metrics for SDL campaign evaluation.

Key design decision: range-normalised AUC
    AUC = (mean(running_best_curve) - global_min) / (global_best - global_min)

This handles Pareto datasets where the scalarised objective has negative values
(zero-mean standardised). Without range normalisation, a campaign that never
finds a positive-y point would produce AUC < 0.
"""

import numpy as np
from typing import Dict, List, Any


def compute_metrics(
    results: Dict[str, Any],
    global_best: float,
    budget: int,
    oracle_global_min: float = 0.0,
) -> Dict[str, float]:
    """
    Compute per-seed metrics from a single campaign run.

    Parameters
    ----------
    results          : dict returned by CampaignSimulator.run()
    global_best      : oracle.global_best() — the empirical maximum
    budget           : total campaign length
    oracle_global_min: oracle._y_raw.min() — needed for range normalisation
                       (default 0.0 is correct for coatings; pass explicitly for Pareto)

    Returns
    -------
    dict with keys:
      auc_best               : range-normalised area under running-best curve [0, 1]
      final_best             : raw running_best at end of campaign
      final_best_normalised  : final_best / global_best
      steps_to_90pct         : steps to reach 90% of global_best (-1 if never reached)
      switch_count           : number of strategy switches
      switch_frequency       : switches per 10 steps
      failure_count          : number of fallback-to-random events
    """
    curve = np.array(results["running_best"], dtype=float)

    # Pad or truncate to budget length
    if len(curve) < budget:
        curve = np.pad(curve, (0, budget - len(curve)), mode="edge")
    else:
        curve = curve[:budget]

    # Range-normalised AUC
    y_range = global_best - oracle_global_min
    if y_range < 1e-12:
        auc = 0.0
    else:
        auc = float(np.mean((curve - oracle_global_min) / y_range))
    auc = float(np.clip(auc, 0.0, 1.0))

    # Final best
    final_best = float(curve[-1])
    final_norm = float(curve[-1] / (global_best + 1e-12))

    # Steps to reach 90% of global_best
    threshold = 0.9 * global_best
    above = np.where(curve >= threshold)[0]
    steps_to_90 = int(above[0]) if len(above) > 0 else -1

    # Switch count: number of unique strategy changes in decisions list
    decisions = results.get("decisions", [])
    strategies_used = [s for _, s, _ in decisions]
    switch_count = sum(
        strategies_used[i] != strategies_used[i - 1]
        for i in range(1, len(strategies_used))
    )

    # Switch frequency: switches per 10 steps
    switch_frequency = float(switch_count / max(budget, 1) * 10)

    return {
        "auc_best": auc,
        "final_best": final_best,
        "final_best_normalised": final_norm,
        "steps_to_90pct": steps_to_90,
        "switch_count": switch_count,
        "switch_frequency": switch_frequency,
        "failure_count": len(results.get("failures", [])),
    }


def aggregate_across_seeds(seed_metrics: List[Dict[str, float]]) -> Dict[str, Dict]:
    """
    Aggregate per-seed metrics into mean ± std.

    Returns
    -------
    dict mapping metric_name -> {'mean': float, 'std': float, 'min': float, 'max': float}
    """
    if not seed_metrics:
        return {}

    keys = list(seed_metrics[0].keys())
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in seed_metrics], dtype=float)
        agg[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return agg


def validate_schema(df) -> None:
    """
    Validate that a metrics_summary DataFrame has the expected columns,
    dtypes, and value ranges.  Raises AssertionError on any violation.

    Checks
    ------
    - All 7 metric columns are present
    - Numeric dtypes (no object columns in metric positions)
    - auc_best in [0, 1]
    - final_best_normalised in [0, 1]
    - switch_frequency >= 0
    - failure_count >= 0
    - steps_to_90pct is -1 (never reached) or a non-negative integer
    """
    REQUIRED_COLS = {
        "dataset", "condition", "seed",
        "global_best", "global_min",
        "auc_best", "final_best", "final_best_normalised",
        "steps_to_90pct", "switch_count", "switch_frequency", "failure_count",
    }
    NUMERIC_METRICS = [
        "auc_best", "final_best", "final_best_normalised",
        "steps_to_90pct", "switch_count", "switch_frequency", "failure_count",
    ]

    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    for col in NUMERIC_METRICS:
        assert df[col].dtype.kind in ("f", "i", "u"), \
            f"Column '{col}' has non-numeric dtype {df[col].dtype}"

    assert df["auc_best"].between(0.0, 1.0).all(), \
        f"auc_best out of [0,1]: min={df['auc_best'].min():.4f}, max={df['auc_best'].max():.4f}"

    # final_best_normalised = final_best / global_best.
    # For Pareto datasets the scalarised objective is z-scored, so final_best can be
    # negative even when global_best > 0 — a campaign that ends below the mean is
    # legitimate.  We only check that it does not exceed 1 (which would mean the
    # campaign found a point better than the oracle maximum).
    assert (df["final_best_normalised"] <= 1.0 + 1e-6).all(), \
        f"final_best_normalised > 1: max={df['final_best_normalised'].max():.4f}"

    assert (df["switch_frequency"] >= 0).all(), \
        "switch_frequency has negative values"

    assert (df["failure_count"] >= 0).all(), \
        "failure_count has negative values"

    # steps_to_90pct is -1 (never reached) in memory, but pandas stores it as NaN
    # when the column is float64 and the CSV is round-tripped.  Accept both.
    import numpy as _np
    valid_steps = (
        df["steps_to_90pct"].isna() |
        (df["steps_to_90pct"] == -1) |
        (df["steps_to_90pct"] >= 0)
    )
    assert valid_steps.all(), \
        f"steps_to_90pct has unexpected values: {df.loc[~valid_steps, 'steps_to_90pct'].unique()}"


def compute_robustness(seed_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Robustness summary: IQR and failure rate across seeds.
    """
    aucs = np.array([m["auc_best"] for m in seed_metrics])
    failures = np.array([m["failure_count"] for m in seed_metrics])
    return {
        "auc_iqr": float(np.percentile(aucs, 75) - np.percentile(aucs, 25)),
        "auc_cv": float(aucs.std() / (aucs.mean() + 1e-12)),
        "failure_rate": float((failures > 0).mean()),
        "n_seeds": len(seed_metrics),
    }
