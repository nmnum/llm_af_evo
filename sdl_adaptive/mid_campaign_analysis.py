"""
mid_campaign_analysis.py — Compute mid-campaign regret conditioned on bad starts.

Filters to seeds where best_norm < threshold at step n_init (these faced a real
search problem). Computes AUC over the mid-campaign window only.

Works with any set of conditions present in results_dir — including EGBO,
novelty_egbo, LLM approaches, and fixed baselines. No hardcoded condition lists.

Usage:
    # Original results (all conditions in one dir):
    python mid_campaign_analysis.py --results_dir results/ --threshold 0.80

    # Shared-seed experiment:
    python mid_campaign_analysis.py --results_dir results_shared/ --threshold 0.80

    # Power run with baselines from a separate dir:
    python mid_campaign_analysis.py \\
        --results_dir results_power/ \\
        --baseline_results_dir results/ \\
        --threshold 0.80
"""

import argparse
import json
import pathlib
import numpy as np
import pandas as pd
from scipy import stats

DATASETS = [
    "coatings", "pareto_20201218", "pareto_20201223",
    "pareto_20210104", "pareto_20210112", "hartmann3", "hartmann6"
]

# Display labels — any condition not listed here gets its raw name
LABELS = {
    "fixed_random":    "Random",
    "fixed_lhs":       "LHS",
    "fixed_ucb_low":   "UCB β=0.2",
    "fixed_ucb_high":  "UCB β=400",
    "fixed_ei":        "EI (fixed)",
    "mock_approach_a": "Mock-A",
    "mock_approach_b": "Mock-B",
    "mock_approach_c": "Mock-C",
    "approach_a":      "LLM-A",
    "approach_b":      "LLM-B",
    "approach_c":      "LLM-C",
    "approach_c_evo":  "LLM-C+Evo",
    "egbo":            "EGBO",
    "novelty_egbo":    "Novelty-EGBO",
    "ada_original":    "ADA orig.",
}

# Condition groups for structured comparisons
FIXED_GROUP   = {"fixed_random", "fixed_lhs", "fixed_ucb_low",
                 "fixed_ucb_high", "fixed_ei", "ada_original"}
MOCK_GROUP    = {"mock_approach_a", "mock_approach_b", "mock_approach_c"}
LLM_GROUP     = {"approach_a", "approach_b", "approach_c", "approach_c_evo"}
EGBO_GROUP    = {"egbo", "novelty_egbo"}


def _label(cond):
    return LABELS.get(cond, cond)


def _load_curves_norm(cond_dir, global_min, y_range):
    p = cond_dir / "curves.npy"
    if not p.exists():
        return None
    curves = np.load(p)
    return np.clip((curves - global_min) / y_range, 0, 1)


def _get_global_stats(ds, *dirs):
    """Get global_best and global_min from first available metrics_summary.csv."""
    for d in dirs:
        if d is None:
            continue
        s = d / "metrics_summary.csv"
        if s.exists():
            df = pd.read_csv(s)
            row = df[df.dataset == ds].head(1)
            if len(row):
                return float(row["global_best"].iloc[0]), float(row["global_min"].iloc[0])
        # Fall back to per-seed metrics
        for cond_dir in (d / ds).iterdir() if (d / ds).exists() else []:
            mp = cond_dir / "metrics_per_seed.json"
            if mp.exists():
                with open(mp) as f:
                    m = json.load(f)[0]
                fn = m.get("final_best_normalised", 0)
                if fn and fn > 0:
                    return m["final_best"] / fn, 0.0
    return None, None


def _discover_conditions(ds, *dirs):
    """Find all conditions with curves.npy in any of the given directories."""
    found = {}  # cond -> path
    for d in dirs:
        if d is None or not (d / ds).exists():
            continue
        for cond_dir in sorted((d / ds).iterdir()):
            if cond_dir.is_dir() and (cond_dir / "curves.npy").exists():
                cond = cond_dir.name
                if cond not in found:
                    found[cond] = cond_dir
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--baseline_results_dir", default=None,
                        help="Separate dir for baseline conditions (optional)")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="best_norm threshold at step n_init (default 0.80)")
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--window_frac", type=float, default=0.5,
                        help="End of mid-campaign window as fraction of budget")
    args = parser.parse_args()

    results_dir  = pathlib.Path(args.results_dir)
    baseline_dir = (pathlib.Path(args.baseline_results_dir)
                    if args.baseline_results_dir else None)

    print(f"Mid-campaign AUC | threshold={args.threshold} | "
          f"window=steps {args.n_init}–{args.window_frac*100:.0f}% of budget")
    if baseline_dir:
        print(f"Primary:  {results_dir}")
        print(f"Baseline: {baseline_dir}")
    print()

    all_rows = []

    for ds in DATASETS:
        global_best, global_min = _get_global_stats(ds, results_dir, baseline_dir)
        if global_best is None:
            continue

        y_range = global_best - global_min
        if y_range < 1e-9:
            continue

        # Discover all available conditions
        cond_dirs = _discover_conditions(ds, results_dir, baseline_dir)
        if not cond_dirs:
            continue

        # Load all condition curves
        cond_curves = {}
        budget = None
        for cond, cdir in cond_dirs.items():
            cn = _load_curves_norm(cdir, global_min, y_range)
            if cn is not None:
                cond_curves[cond] = cn
                if budget is None:
                    budget = cn.shape[1]

        if not cond_curves or budget is None:
            continue

        window_end = max(args.n_init + 1, int(args.window_frac * budget))

        # Identify qualifying seeds using the condition with the most seeds
        # (for shared-seed experiments all conditions should have identical n_seeds)
        ref_cond = max(cond_curves, key=lambda c: cond_curves[c].shape[0])
        ref_curves = cond_curves[ref_cond]
        qualifying = [i for i in range(len(ref_curves))
                      if ref_curves[i, args.n_init] < args.threshold]

        print(f"{'='*60}")
        print(f"Dataset: {ds}  (budget={budget})")
        print(f"  Qualifying seeds (best_norm < {args.threshold} "
              f"at step {args.n_init}): {len(qualifying)} / {len(ref_curves)}")

        if len(qualifying) < 3:
            print(f"  Too few qualifying seeds — try higher --threshold")
            continue

        # Compute mid-campaign AUC for each condition on qualifying seeds
        cond_aucs = {}
        for cond, cn in cond_curves.items():
            q = [i for i in qualifying if i < len(cn)]
            if not q:
                continue
            aucs = cn[q, args.n_init:window_end].mean(axis=1)
            cond_aucs[cond] = aucs

        # Sort by mean AUC descending
        sorted_conds = sorted(cond_aucs, key=lambda c: -cond_aucs[c].mean())

        print(f"\n  {'Condition':<22} {'n':>4} {'mid-AUC':>9} {'±SD':>7}  group")
        print(f"  {'-'*55}")
        for cond in sorted_conds:
            aucs = cond_aucs[cond]
            grp = ("fixed" if cond in FIXED_GROUP else
                   "mock"  if cond in MOCK_GROUP  else
                   "LLM"   if cond in LLM_GROUP   else
                   "EGBO"  if cond in EGBO_GROUP  else "other")
            print(f"  {_label(cond):<22} {len(aucs):>4} {aucs.mean():>9.3f} "
                  f"{aucs.std():>7.3f}  {grp}")
            all_rows.append({
                "dataset": ds, "condition": cond, "label": _label(cond),
                "n_qualifying": len(aucs),
                "mid_auc_mean": float(aucs.mean()),
                "mid_auc_std":  float(aucs.std()),
                "threshold": args.threshold,
            })

        # ── Statistical tests ─────────────────────────────────────────────
        print(f"\n  Statistical tests (two-sample t-test on qualifying seeds):")

        # 1. Each condition vs the best fixed baseline
        fixed_present = {c: cond_aucs[c] for c in cond_aucs if c in FIXED_GROUP}
        if fixed_present:
            best_fixed_name = max(fixed_present, key=lambda c: fixed_present[c].mean())
            best_fixed_aucs = fixed_present[best_fixed_name]
            print(f"  Best fixed baseline: {_label(best_fixed_name)} "
                  f"(mean={best_fixed_aucs.mean():.3f})")
            for cond in sorted_conds:
                if cond in FIXED_GROUP:
                    continue
                aucs = cond_aucs[cond]
                n = min(len(aucs), len(best_fixed_aucs))
                t, p = stats.ttest_ind(aucs[:n], best_fixed_aucs[:n])
                delta = aucs.mean() - best_fixed_aucs.mean()
                sig = " *" if p < 0.05 else "  "
                print(f"    {_label(cond):<22} vs {_label(best_fixed_name):<12} "
                      f"Δ={delta:+.3f} p={p:.3f}{sig}")

        # 2. EGBO variants vs LLM-C (the key comparison)
        egbo_present = {c: cond_aucs[c] for c in cond_aucs if c in EGBO_GROUP}
        llm_present  = {c: cond_aucs[c] for c in cond_aucs if c in LLM_GROUP}
        if egbo_present and llm_present:
            print(f"\n  EGBO vs LLM approaches:")
            for ec, ea in sorted(egbo_present.items(),
                                  key=lambda x: -x[1].mean()):
                for lc, la in sorted(llm_present.items(),
                                      key=lambda x: -x[1].mean()):
                    n = min(len(ea), len(la))
                    t, p = stats.ttest_ind(ea[:n], la[:n])
                    delta = ea.mean() - la.mean()
                    sig = " *" if p < 0.05 else "  "
                    print(f"    {_label(ec):<22} vs {_label(lc):<14} "
                          f"Δ={delta:+.3f} p={p:.3f}{sig}")

        # 3. novelty_egbo vs egbo (within-EGBO)
        if "novelty_egbo" in cond_aucs and "egbo" in cond_aucs:
            na, ea = cond_aucs["novelty_egbo"], cond_aucs["egbo"]
            n = min(len(na), len(ea))
            t, p = stats.ttest_ind(na[:n], ea[:n])
            delta = na.mean() - ea.mean()
            sig = " *" if p < 0.05 else "  "
            print(f"\n  Within-EGBO:")
            print(f"    Novelty-EGBO vs EGBO           "
                  f"Δ={delta:+.3f} p={p:.3f}{sig}")

        # 4. LLM-C+Evo vs LLM-C (effect of adding evolutionary candidates)
        if "approach_c_evo" in cond_aucs and "approach_c" in cond_aucs:
            ae, ac = cond_aucs["approach_c_evo"], cond_aucs["approach_c"]
            n = min(len(ae), len(ac))
            t, p = stats.ttest_ind(ae[:n], ac[:n])
            delta = ae.mean() - ac.mean()
            sig = " *" if p < 0.05 else "  "
            print(f"\n  Effect of evolutionary candidates on LLM-C:")
            print(f"    LLM-C+Evo vs LLM-C             "
                  f"Δ={delta:+.3f} p={p:.3f}{sig}")

        print()

    # Save and summarise
    out_df = pd.DataFrame(all_rows)
    out_path = results_dir / "mid_campaign_conditioned.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if len(out_df):
        print(f"\n{'='*60}")
        print("SUMMARY: Mean mid-campaign AUC (qualifying seeds, all datasets)")
        print(f"{'='*60}")
        summary = (out_df.groupby("label")["mid_auc_mean"]
                   .mean().sort_values(ascending=False))
        grp_map = {_label(c): (
            "EGBO"  if c in EGBO_GROUP  else
            "LLM"   if c in LLM_GROUP   else
            "mock"  if c in MOCK_GROUP  else
            "fixed" if c in FIXED_GROUP else "other")
            for c in LABELS}
        for label, mean in summary.items():
            grp = grp_map.get(label, "other")
            marker = " ◄" if grp in ("LLM", "EGBO") else ""
            print(f"  {label:<22} {mean:.3f}  [{grp}]{marker}")


if __name__ == "__main__":
    main()
