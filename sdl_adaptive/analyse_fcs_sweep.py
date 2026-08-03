"""
analyse_fcs_sweep.py — Analyse first_control_step sweep results.

For each (dataset, condition, fcs) triple, computes:
  - Full-campaign AUC
  - Conditioned mid-campaign AUC (bad-start seeds)
  - Trust diagnostic outcome at first_control_step

Key comparison: does routing at fcs=20 beat fcs=5 on structured landscapes?
Is there a crossover where delayed routing becomes better?

Usage:
    python analyse_fcs_sweep.py --sweep_dir results_fcs_sweep/
"""

import argparse
import pathlib
import json
import numpy as np
import pandas as pd
from scipy import stats

STRUCTURED = {"coatings", "pareto_20210112", "hartmann3", "hartmann6"}
FLAT       = {"pareto_20201218", "pareto_20201223"}


def load_condition(results_dir: pathlib.Path, ds: str, cond: str):
    p = results_dir / ds / cond / "curves.npy"
    if not p.exists():
        return None
    return np.load(p)


def conditioned_auc(curves, global_best, n_init, threshold=0.80, window_frac=0.5):
    """Mid-campaign AUC on seeds where best_norm < threshold at n_init."""
    budget = curves.shape[1]
    window_end = max(n_init + 1, int(window_frac * budget))
    norms = curves / global_best
    qualifying = [i for i in range(len(norms))
                  if norms[i, n_init] < threshold]
    if not qualifying:
        return None, qualifying
    aucs = norms[qualifying, n_init:window_end].mean(axis=1)
    return aucs, qualifying


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default="results_fcs_sweep")
    parser.add_argument("--n_init",    type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    sweep_dir = pathlib.Path(args.sweep_dir)
    fcs_values = sorted([
        int(d.name.replace("fcs",""))
        for d in sweep_dir.iterdir()
        if d.is_dir() and d.name.startswith("fcs")
    ])

    if not fcs_values:
        print(f"No fcs* directories found in {sweep_dir}")
        return

    print(f"First-control-step sweep: {fcs_values}")
    print()

    rows = []
    for fcs in fcs_values:
        fcs_dir = sweep_dir / f"fcs{fcs}"
        summary = fcs_dir / "metrics_summary.csv"
        if not summary.exists():
            print(f"Missing: {summary}")
            continue

        df = pd.read_csv(summary)
        for ds in df.dataset.unique():
            ds_sub = df[df.dataset == ds]
            gb = float(ds_sub.global_best.iloc[0])

            for cond in ds_sub.condition.unique():
                curves = load_condition(fcs_dir, ds, cond)
                if curves is None:
                    continue

                # Full-campaign AUC
                full_auc = float((curves / gb).mean())

                # Conditioned mid-campaign AUC
                cond_aucs, q_seeds = conditioned_auc(
                    curves, gb, args.n_init, args.threshold)
                cond_auc = float(cond_aucs.mean()) if cond_aucs is not None else None

                rows.append({
                    "fcs": fcs, "dataset": ds, "condition": cond,
                    "full_auc": full_auc, "cond_auc": cond_auc,
                    "n_qualifying": len(q_seeds),
                    "regime": "structured" if ds in STRUCTURED else "flat",
                })

    df_all = pd.DataFrame(rows)
    df_all.to_csv(sweep_dir / "sweep_summary.csv", index=False)
    print(f"Saved: {sweep_dir}/sweep_summary.csv")

    # ── Key comparison: AUC vs first_control_step ──────────────────────────
    print()
    print("="*70)
    print("FULL-CAMPAIGN AUC vs FIRST_CONTROL_STEP")
    print("="*70)

    for ds in df_all.dataset.unique():
        regime = "structured" if ds in STRUCTURED else "flat"
        print(f"\n{ds} [{regime}]:")
        ds_df = df_all[df_all.dataset == ds]
        conds = [c for c in ["egbo","rule_router","approach_d","fixed_lhs","fixed_random"]
                 if c in ds_df.condition.values]
        print(f"  {'Condition':<16}", end="")
        for fcs in fcs_values:
            print(f"  fcs={fcs:>2}", end="")
        print()
        print(f"  {'-'*60}")
        for cond in conds:
            print(f"  {cond:<16}", end="")
            for fcs in fcs_values:
                row = ds_df[(ds_df.fcs==fcs) & (ds_df.condition==cond)]
                val = f"{row.full_auc.values[0]:.3f}" if len(row) else "  — "
                print(f"  {val:>6}", end="")
            print()

    # ── Crossover analysis ─────────────────────────────────────────────────
    print()
    print("="*70)
    print("CROSSOVER ANALYSIS: does delayed routing beat immediate routing?")
    print("Comparison: fcs=20 vs fcs=5 for approach_d on each dataset")
    print("="*70)
    for ds in df_all.dataset.unique():
        ds_df = df_all[df_all.dataset == ds]
        for cond in ["approach_d", "rule_router"]:
            fcs5  = ds_df[(ds_df.fcs==5)  & (ds_df.condition==cond)]["full_auc"]
            fcs20 = ds_df[(ds_df.fcs==20) & (ds_df.condition==cond)]["full_auc"]
            if len(fcs5) and len(fcs20):
                delta = float(fcs20.values[0] - fcs5.values[0])
                better = "delayed BETTER" if delta > 0.005 else \
                         "immediate BETTER" if delta < -0.005 else "TIED"
                print(f"  {ds} {cond}: fcs5={fcs5.values[0]:.3f} "
                      f"fcs20={fcs20.values[0]:.3f} Δ={delta:+.3f} → {better}")

    # ── Budget cost of delay ───────────────────────────────────────────────
    print()
    print("="*70)
    print("BUDGET COST OF DELAY (vs always-EGBO at fcs=5)")
    print("="*70)
    egbo_fcs5 = df_all[(df_all.fcs==5) & (df_all.condition=="egbo")]
    for ds in df_all.dataset.unique():
        ref = egbo_fcs5[egbo_fcs5.dataset==ds]["full_auc"]
        if not len(ref):
            continue
        ref_val = float(ref.values[0])
        print(f"\n  {ds} — always-EGBO (fcs=5): {ref_val:.3f}")
        for fcs in fcs_values[1:]:
            for cond in ["approach_d"]:
                row = df_all[(df_all.fcs==fcs) & (df_all.condition==cond)
                             & (df_all.dataset==ds)]
                if len(row):
                    v = float(row.full_auc.values[0])
                    print(f"    approach_d fcs={fcs}: {v:.3f}  "
                          f"Δ vs always-EGBO = {v-ref_val:+.3f}")


if __name__ == "__main__":
    main()
