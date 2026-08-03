"""
run_sensitivity.py — Novelty weight sensitivity sweep + isolated warm-start condition.

Adds two things missing from Phase 1:
  1. mo_ls_egbo: LLM warm-start + real EGBO (qLogNEHVI + U-NSGA-III), NO novelty
     selection. Isolates the pure LLM warm-start effect from the novelty penalty.
  2. Novelty weight sweep: w_nov in {0.0, 0.1, 0.15, 0.2, 0.3} for mo_ls_na_egbo,
     to find where novelty selection stops hurting.

Also fixes the exp_to_threshold metric: uses 70% instead of 90% (which was
unreachable at budget=30).

Usage:
    python run_sensitivity.py --mock_llm --n_seeds 15 --budget 30 \
        --out_dir /mnt/results/benchmark_sensitivity
"""

import argparse
import json
import pathlib
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from excipient_oracle_mo import MultiObjectiveExcipientOracle
from excipient_oracle import (
    AMINO_ACIDS, SUGARS, SURFACTANTS,
    AA_LIST, SUGAR_LIST, SF_LIST, FEATURE_DIM,
    formulation_to_vector, vector_to_formulation,
    PROTEIN_PROFILES,
)
from excipient_campaign_mo import (
    strategy_mo_random, strategy_mo_egbo, strategy_mo_egbo_real,
    strategy_mo_llm, run_mo_campaign, make_shared_inits,
    pareto_front_of, OBJECTIVE_DIRECTIONS, OBJECTIVE_NAMES,
    MO_PRIORS,
)
from strategy_ls_na_egbo import (
    strategy_mo_egbo_novelty, strategy_mo_ls_na_egbo,
    run_ls_na_egbo_campaign,
)
from strategy_llm_candidate_gen import strategy_mo_llm_candidate_gen


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_hv(Y_obs, ref_point):
    try:
        from pymoo.indicators.hv import HV
        Y = Y_obs.copy()
        for j, d in enumerate(OBJECTIVE_DIRECTIONS):
            if d == "max":
                Y[:, j] = -Y[:, j]
        pf_idx = pareto_front_of(Y_obs)
        Y_pf = Y[pf_idx]
        return float(HV(ref_point=ref_point)(Y_pf))
    except Exception:
        return float("nan")


def compute_igd(Y_obs, true_pareto, ref_point):
    try:
        from pymoo.indicators.igd import IGD
        Y = Y_obs.copy()
        for j, d in enumerate(OBJECTIVE_DIRECTIONS):
            if d == "max":
                Y[:, j] = -Y[:, j]
        pf_idx = pareto_front_of(Y_obs)
        Y_pf = Y[pf_idx]
        return float(IGD(true_pareto)(Y_pf))
    except Exception:
        return float("nan")


def experiments_to_threshold(hv_trajectory, n_init, batch_size, threshold_frac, max_hv):
    """Fixed: uses configurable threshold (default 70% instead of unreachable 90%)."""
    threshold = threshold_frac * max_hv
    n_batches = len(hv_trajectory)
    for b, hv in enumerate(hv_trajectory):
        if hv >= threshold:
            return n_init + (b + 1) * batch_size
    return n_init + n_batches * batch_size


def count_stagnant_batches(hv_trajectory):
    if len(hv_trajectory) < 2:
        return 0
    stagnant = 0
    for i in range(1, len(hv_trajectory)):
        if hv_trajectory[i] <= hv_trajectory[i-1] + 1e-6:
            stagnant += 1
    return stagnant


# ── Checkpoint management ──────────────────────────────────────────────────────

CKPT_COLS = [
    "phase", "protein", "prior_level", "condition", "seed",
    "budget", "n_init", "final_hv", "final_pf_size",
    "igd", "exp_to_70pct_hv", "exp_to_80pct_hv", "stagnant_batches", "max_hv", "n_obs",
    "w_nov",
]


def load_checkpoint(ckpt_path):
    if pathlib.Path(ckpt_path).exists():
        try:
            df = pd.read_csv(ckpt_path)
            print(f"  Loaded checkpoint: {len(df)} rows from {ckpt_path}")
            return df
        except Exception as e:
            print(f"  Warning: could not load checkpoint: {e}")
    return pd.DataFrame(columns=CKPT_COLS)


def save_checkpoint(df, ckpt_path):
    df.to_csv(str(ckpt_path), index=False)


def is_completed(ckpt_df, phase, protein, prior, condition, seed, w_nov=None):
    if len(ckpt_df) == 0:
        return False
    mask = (
        (ckpt_df["phase"] == phase) &
        (ckpt_df["protein"] == protein) &
        (ckpt_df["prior_level"] == prior) &
        (ckpt_df["condition"] == condition) &
        (ckpt_df["seed"] == seed)
    )
    if w_nov is not None and "w_nov" in ckpt_df.columns:
        mask = mask & (ckpt_df["w_nov"] == w_nov)
    return mask.any()


# ── Main sensitivity runner ───────────────────────────────────────────────────

def run_sensitivity(args):
    """
    Run:
    1. mo_ls_egbo (LLM warm-start + real EGBO, no novelty) — isolates warm-start
    2. Novelty weight sweep for mo_ls_na_egbo: w_nov in {0.0, 0.1, 0.15, 0.2, 0.3}
    """
    print(f"\n{'='*60}")
    print(f"SENSITIVITY ANALYSIS: novelty weight + isolated warm-start")
    print(f"{'='*60}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sensitivity_raw_results.csv"

    ckpt_df = load_checkpoint(ckpt_path)
    results = ckpt_df.to_dict("records")

    novelty_weights = [0.0, 0.1, 0.15, 0.2, 0.3]

    # Count total combos
    # mo_ls_egbo: 4 combos × 15 seeds = 60
    # mo_ls_na_egbo sweep: 4 combos × 15 seeds × 5 weights = 300
    # mo_egbo_novelty sweep (no LLM): 4 combos × 15 seeds × 5 weights = 300
    total = len(args.proteins) * len(args.priors) * (1 + len(novelty_weights) * 2) * args.n_seeds
    done = len(results)
    print(f"  Progress: {done}/{total} combos completed")

    for protein in args.proteins:
        for prior_level in args.priors:
            print(f"\n  Protein: {protein}, Prior: {prior_level}")

            oracle = MultiObjectiveExcipientOracle(
                protein=protein,
                tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
                seed=42,
            )
            disc = oracle.make_discrete_oracle(n_samples=500, seed=42)

            Y_all = disc._Y_raw.copy()
            Y_all_min = Y_all.copy()
            for j, d in enumerate(OBJECTIVE_DIRECTIONS):
                if d == "max":
                    Y_all_min[:, j] = -Y_all_min[:, j]
            pf_true_idx = pareto_front_of(Y_all)
            true_pareto = Y_all_min[pf_true_idx]
            ref_point = (Y_all_min.max(axis=0) +
                        0.1 * (Y_all_min.max(axis=0) - Y_all_min.min(axis=0) + 1e-9))
            max_hv = compute_hv(Y_all, ref_point)

            shared_inits = make_shared_inits(disc, args.n_seeds, args.n_init, rng_seed=42)

            prior_key = ("agg_L1" if prior_level == "L1" and "aggregation" in protein
                         else "oxid_L1" if prior_level == "L1" and "oxidation" in protein
                         else "wrong" if prior_level == "wrong" else "blank")
            prior_text = MO_PRIORS.get(prior_key, MO_PRIORS["blank"])

            # ── Condition 1: mo_ls_egbo (LLM warm-start + real EGBO, NO novelty) ──
            # Uses strategy_mo_egbo_real (greedy top-k, no novelty) but with
            # LLM warm-start init instead of shared random init
            cond_name = "mo_ls_egbo"
            t0 = time.time()
            ran = 0
            skipped = 0
            for seed_idx in range(args.n_seeds):
                if is_completed(ckpt_df, 0, protein, prior_level, cond_name, seed_idx):
                    skipped += 1
                    continue

                disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                result = run_ls_na_egbo_campaign(
                    disc_seed, budget=args.budget,
                    strategy_fn=strategy_mo_egbo_real, strategy_kwargs={},
                    batch_size=args.batch_size, seed=seed_idx,
                    n_init=args.n_init, n_propose=25,
                    protein=protein, prior_level=prior_level,
                    model=args.model, mock_llm=args.mock_llm,
                )

                final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
                final_pf = len(pareto_front_of(result["Y_obs"]))
                igd = compute_igd(result["Y_obs"], true_pareto, ref_point)
                exp70 = experiments_to_threshold(result["hv_trajectory"], args.n_init, args.batch_size, 0.70, max_hv)
                exp80 = experiments_to_threshold(result["hv_trajectory"], args.n_init, args.batch_size, 0.80, max_hv)
                stag = count_stagnant_batches(result["hv_trajectory"])

                row = {
                    "phase": 0, "protein": protein, "prior_level": prior_level,
                    "condition": cond_name, "seed": seed_idx,
                    "budget": args.budget, "n_init": args.n_init,
                    "final_hv": final_hv, "final_pf_size": final_pf,
                    "igd": igd, "exp_to_70pct_hv": exp70, "exp_to_80pct_hv": exp80,
                    "stagnant_batches": stag, "max_hv": max_hv,
                    "n_obs": len(result["Y_obs"]), "w_nov": -1,
                }
                results.append(row)
                ran += 1
                df_tmp = pd.DataFrame(results, columns=CKPT_COLS)
                save_checkpoint(df_tmp, ckpt_path)

            elapsed = time.time() - t0
            status = f"ran {ran}" if ran > 0 else "all cached"
            if skipped > 0: status += f", skipped {skipped}"
            print(f"    {cond_name}: {status} ({elapsed:.0f}s)")

            # ── Condition 2: mo_ls_na_egbo novelty weight sweep ──
            for w_nov in novelty_weights:
                w_acq = 1.0 - w_nov
                cond_name = f"mo_ls_na_egbo_w{w_nov:.2f}"
                t0 = time.time()
                ran = 0
                skipped = 0
                for seed_idx in range(args.n_seeds):
                    if is_completed(ckpt_df, 0, protein, prior_level, cond_name, seed_idx, w_nov=w_nov):
                        skipped += 1
                        continue

                    disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                    result = run_ls_na_egbo_campaign(
                        disc_seed, budget=args.budget,
                        strategy_fn=strategy_mo_ls_na_egbo,
                        strategy_kwargs={"w_acq": w_acq, "w_nov": w_nov},
                        batch_size=args.batch_size, seed=seed_idx,
                        n_init=args.n_init, n_propose=25,
                        protein=protein, prior_level=prior_level,
                        model=args.model, mock_llm=args.mock_llm,
                    )

                    final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
                    final_pf = len(pareto_front_of(result["Y_obs"]))
                    igd = compute_igd(result["Y_obs"], true_pareto, ref_point)
                    exp70 = experiments_to_threshold(result["hv_trajectory"], args.n_init, args.batch_size, 0.70, max_hv)
                    exp80 = experiments_to_threshold(result["hv_trajectory"], args.n_init, args.batch_size, 0.80, max_hv)
                    stag = count_stagnant_batches(result["hv_trajectory"])

                    row = {
                        "phase": 0, "protein": protein, "prior_level": prior_level,
                        "condition": cond_name, "seed": seed_idx,
                        "budget": args.budget, "n_init": args.n_init,
                        "final_hv": final_hv, "final_pf_size": final_pf,
                        "igd": igd, "exp_to_70pct_hv": exp70, "exp_to_80pct_hv": exp80,
                        "stagnant_batches": stag, "max_hv": max_hv,
                        "n_obs": len(result["Y_obs"]), "w_nov": w_nov,
                    }
                    results.append(row)
                    ran += 1
                    df_tmp = pd.DataFrame(results, columns=CKPT_COLS)
                    save_checkpoint(df_tmp, ckpt_path)

                elapsed = time.time() - t0
                status = f"ran {ran}" if ran > 0 else "all cached"
                if skipped > 0: status += f", skipped {skipped}"
                print(f"    {cond_name}: {status} ({elapsed:.0f}s)")

            # ── Condition 3: mo_egbo_novelty weight sweep (no LLM, shared init) ──
            for w_nov in novelty_weights:
                w_acq = 1.0 - w_nov
                cond_name = f"mo_egbo_novelty_w{w_nov:.2f}"
                t0 = time.time()
                ran = 0
                skipped = 0
                for seed_idx in range(args.n_seeds):
                    if is_completed(ckpt_df, 0, protein, prior_level, cond_name, seed_idx, w_nov=w_nov):
                        skipped += 1
                        continue

                    disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                    X_init, Y_init = shared_inits[seed_idx]
                    result = run_mo_campaign(
                        disc_seed, X_init, Y_init, args.budget,
                        strategy_mo_egbo_novelty,
                        {"w_acq": w_acq, "w_nov": w_nov},
                        batch_size=args.batch_size, seed=seed_idx,
                    )

                    final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
                    final_pf = len(pareto_front_of(result["Y_obs"]))
                    igd = compute_igd(result["Y_obs"], true_pareto, ref_point)
                    exp70 = experiments_to_threshold(result["hv_trajectory"], args.n_init, args.batch_size, 0.70, max_hv)
                    exp80 = experiments_to_threshold(result["hv_trajectory"], args.n_init, args.batch_size, 0.80, max_hv)
                    stag = count_stagnant_batches(result["hv_trajectory"])

                    row = {
                        "phase": 0, "protein": protein, "prior_level": prior_level,
                        "condition": cond_name, "seed": seed_idx,
                        "budget": args.budget, "n_init": args.n_init,
                        "final_hv": final_hv, "final_pf_size": final_pf,
                        "igd": igd, "exp_to_70pct_hv": exp70, "exp_to_80pct_hv": exp80,
                        "stagnant_batches": stag, "max_hv": max_hv,
                        "n_obs": len(result["Y_obs"]), "w_nov": w_nov,
                    }
                    results.append(row)
                    ran += 1
                    df_tmp = pd.DataFrame(results, columns=CKPT_COLS)
                    save_checkpoint(df_tmp, ckpt_path)

                elapsed = time.time() - t0
                status = f"ran {ran}" if ran > 0 else "all cached"
                if skipped > 0: status += f", skipped {skipped}"
                print(f"    {cond_name}: {status} ({elapsed:.0f}s)")

    df = pd.DataFrame(results, columns=CKPT_COLS)
    return df


def main():
    parser = argparse.ArgumentParser(description="Novelty weight sensitivity + isolated warm-start")
    parser.add_argument("--n_seeds", type=int, default=15)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--proteins", nargs="+", default=["mAb_aggregation", "mAb_oxidation"])
    parser.add_argument("--priors", nargs="+", default=["L1", "blank"])
    parser.add_argument("--out_dir", default="/mnt/results/benchmark_sensitivity")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sensitivity analysis configuration:")
    print(f"  Seeds: {args.n_seeds}, Budget: {args.budget}")
    print(f"  Proteins: {args.proteins}, Priors: {args.priors}")
    print(f"  Novelty weights: [0.0, 0.1, 0.15, 0.2, 0.3]")
    print(f"  Mock LLM: {args.mock_llm}")

    t_start = time.time()
    df = run_sensitivity(args)
    elapsed = time.time() - t_start
    print(f"\n\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    ckpt_path = out_dir / "sensitivity_raw_results.csv"
    df.to_csv(ckpt_path, index=False)
    print(f"Saved: {ckpt_path}")

    # Summary
    summary = df.groupby(["protein", "prior_level", "condition"]).agg(
        hv_mean=("final_hv", "mean"),
        hv_std=("final_hv", "std"),
        igd_mean=("igd", "mean"),
        exp70_mean=("exp_to_70pct_hv", "mean"),
        exp80_mean=("exp_to_80pct_hv", "mean"),
        stag_mean=("stagnant_batches", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)
    print(f"Saved: {out_dir / 'sensitivity_summary.csv'}")

    # Print summary
    print(f"\n{'='*90}")
    print(f"SENSITIVITY RESULTS")
    print(f"{'='*90}")
    for (protein, prior), group in summary.groupby(["protein", "prior_level"]):
        print(f"\n  Protein: {protein}, Prior: {prior}")
        print(f"  {'Condition':<30} {'HV mean±std':>15} {'IGD':>6} {'Exp→70%':>8} {'Exp→80%':>8} {'Stag':>5}")
        print(f"  {'-'*85}")
        for _, row in group.iterrows():
            print(f"  {row['condition']:<30} {row['hv_mean']:>8.0f}±{row['hv_std']:<5.0f} "
                  f"{row['igd_mean']:>6.1f} {row['exp70_mean']:>8.1f} {row['exp80_mean']:>8.1f} "
                  f"{row['stag_mean']:>5.1f}")


if __name__ == "__main__":
    main()
