"""
run_benchmark.py — Unified benchmark runner for LLM-augmented formulation optimization.

Runs all 7 conditions across 3 phases, collects metrics, runs statistical tests,
and outputs CSV + figures.

Conditions:
  mo_random       — Random search (floor baseline)
  mo_egbo         — Lightweight EGBO (per-objective GP + Pareto)
  mo_egbo_real    — Real EGBO (qLogNEHVI + U-NSGA-III)
  mo_egbo_novelty — EGBO + novelty-aware selection (Aqeeli et al.)
  mo_ls_na_egbo   — LLM warm-start + novelty-aware EGBO (recommended)
  mo_llm_candidate_gen — LLM-as-candidate-generator (LLM every batch + qLogNEHVI scoring; NOT the literature LABO method, see strategy_llm_candidate_gen.py docstring)
  mo_llm_existing — Existing trust-weighted LLM/GP mixing

Phases:
  Phase 1: Reduced formulation space (fast iteration, novelty weight tuning)
  Phase 2: Full 14-excipient space (real problem)
  Phase 3: Ada coatings (cross-domain generalisability)

Usage:
    # Phase 1, mock LLM, 5 seeds (quick test)
    python run_benchmark.py --phase 1 --mock_llm --n_seeds 5 --budget 30

    # Phase 2, real LLM, 15 seeds
    python run_benchmark.py --phase 2 --model qwen2.5:72b-instruct --n_seeds 15 --budget 40

    # Phase 3, Ada coatings
    python run_benchmark.py --phase 3 --mock_llm --n_seeds 20
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

from excipient_oracle_mo import MultiObjectiveExcipientOracle, TM_RANGE, KD_RANGE, VISC_RANGE
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
    """Compute hypervolume of the Pareto front from observations."""
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
    """Compute IGD (inverted generational distance) to the true Pareto front."""
    try:
        from pymoo.indicators.igd import IGD
        Y = Y_obs.copy()
        for j, d in enumerate(OBJECTIVE_DIRECTIONS):
            if d == "max":
                Y[:, j] = -Y[:, j]
        pf_idx = pareto_front_of(Y_obs)
        Y_pf = Y[pf_idx]
        # true_pareto is already in minimisation convention
        return float(IGD(true_pareto)(Y_pf))
    except Exception:
        return float("nan")


def experiments_to_threshold(hv_trajectory, n_init, batch_size, threshold_frac, max_hv):
    """
    Estimate how many experiments are needed to reach threshold_frac of max_hv.
    Returns the experiment count, or budget if never reached.
    """
    threshold = threshold_frac * max_hv
    # HV trajectory is per-batch; convert to per-experiment
    n_batches = len(hv_trajectory)
    for b, hv in enumerate(hv_trajectory):
        if hv >= threshold:
            return n_init + (b + 1) * batch_size
    return n_init + n_batches * batch_size  # never reached → full budget


def count_stagnant_batches(hv_trajectory):
    """Count batches with zero or negative HV improvement."""
    if len(hv_trajectory) < 2:
        return 0
    stagnant = 0
    for i in range(1, len(hv_trajectory)):
        if hv_trajectory[i] <= hv_trajectory[i-1] + 1e-6:
            stagnant += 1
    return stagnant


# ── Phase 1: Reduced formulation space ─────────────────────────────────────────

def run_phase1(args):
    """
    Reduced space: 1 AA × 1 sugar × 1 surfactant ± EDTA.
    Fast iteration for architecture validation and novelty weight tuning.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 1: Reduced formulation space")
    print(f"{'='*60}")

    # Use the full oracle but with a smaller discrete pool to simulate
    # the reduced space. The oracle's mechanism logic still applies.
    results = []

    for protein in args.proteins:
        for prior_level in args.priors:
            print(f"\n  Protein: {protein}, Prior: {prior_level}")

            oracle = MultiObjectiveExcipientOracle(
                protein=protein,
                tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
                seed=42,
            )
            disc = oracle.make_discrete_oracle(n_samples=500, seed=42)

            # Compute ground truth Pareto front for IGD
            Y_all = disc._Y_raw.copy()
            Y_all_min = Y_all.copy()
            for j, d in enumerate(OBJECTIVE_DIRECTIONS):
                if d == "max":
                    Y_all_min[:, j] = -Y_all_min[:, j]
            pf_true_idx = pareto_front_of(Y_all)
            true_pareto = Y_all_min[pf_true_idx]

            # Fixed reference point
            ref_point = (Y_all_min.max(axis=0) +
                        0.1 * (Y_all_min.max(axis=0) - Y_all_min.min(axis=0) + 1e-9))
            max_hv = compute_hv(Y_all, ref_point)

            # Shared random inits for non-LLM conditions
            shared_inits = make_shared_inits(disc, args.n_seeds, args.n_init, rng_seed=42)

            prior_key = ("agg_L1" if prior_level == "L1" and "aggregation" in protein
                         else "oxid_L1" if prior_level == "L1" and "oxidation" in protein
                         else "blank")
            prior_text = MO_PRIORS.get(prior_key, MO_PRIORS["blank"])

            conditions = {
                "mo_random": (strategy_mo_random, {}, False),       # (fn, kwargs, is_llm_warmstart)
                "mo_egbo": (strategy_mo_egbo, {}, False),
                "mo_egbo_real": (strategy_mo_egbo_real, {}, False),
                "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}, False),
                "mo_ls_na_egbo": (strategy_mo_ls_na_egbo,
                                  {"w_acq": args.w_acq, "w_nov": args.w_nov}, True),
                "mo_llm_candidate_gen": (strategy_mo_llm_candidate_gen,
                                {"prior_text": prior_text, "mock_llm": args.mock_llm,
                                 "model": args.model, "n_llm_candidates": 20}, False),
            }

            for cond_name, (fn, kwargs, is_warmstart) in conditions.items():
                t0 = time.time()
                print(f"    {cond_name}...", end="", flush=True)

                for seed_idx in range(args.n_seeds):
                    # Fresh oracle per seed for LLM warm-start conditions
                    # (they generate their own init points)
                    if is_warmstart:
                        disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                        result = run_ls_na_egbo_campaign(
                            disc_seed, budget=args.budget,
                            strategy_fn=fn, strategy_kwargs=kwargs,
                            batch_size=args.batch_size, seed=seed_idx,
                            n_init=args.n_init, n_propose=25,
                            protein=protein, prior_level=prior_level,
                            model=args.model, mock_llm=args.mock_llm,
                        )
                    else:
                        disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                        X_init, Y_init = shared_inits[seed_idx]
                        result = run_mo_campaign(
                            disc_seed, X_init, Y_init, args.budget,
                            fn, kwargs, batch_size=args.batch_size, seed=seed_idx,
                        )

                    final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
                    final_pf = len(pareto_front_of(result["Y_obs"]))
                    igd = compute_igd(result["Y_obs"], true_pareto, ref_point)
                    exp_to_90 = experiments_to_threshold(
                        result["hv_trajectory"], args.n_init, args.batch_size,
                        0.9, max_hv)
                    stag = count_stagnant_batches(result["hv_trajectory"])

                    results.append({
                        "phase": 1, "protein": protein, "prior_level": prior_level,
                        "condition": cond_name, "seed": seed_idx,
                        "budget": args.budget, "n_init": args.n_init,
                        "final_hv": final_hv, "final_pf_size": final_pf,
                        "igd": igd, "exp_to_90pct_hv": exp_to_90,
                        "stagnant_batches": stag,
                        "max_hv": max_hv,
                        "n_obs": len(result["Y_obs"]),
                    })
                    print(".", end="", flush=True)
                elapsed = time.time() - t0
                print(f" ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    return df


# ── Phase 2: Full 14-excipient space ───────────────────────────────────────────

def run_phase2(args):
    """
    Full excipient catalogue (16D encoded). The real problem.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full 14-excipient space (16D)")
    print(f"{'='*60}")

    results = []

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

            conditions = {
                "mo_random": (strategy_mo_random, {}, False),
                "mo_egbo": (strategy_mo_egbo, {}, False),
                "mo_egbo_real": (strategy_mo_egbo_real, {}, False),
                "mo_egbo_novelty": (strategy_mo_egbo_novelty, {}, False),
                "mo_ls_na_egbo": (strategy_mo_ls_na_egbo,
                                  {"w_acq": args.w_acq, "w_nov": args.w_nov}, True),
                "mo_llm_candidate_gen": (strategy_mo_llm_candidate_gen,
                                {"prior_text": prior_text, "mock_llm": args.mock_llm,
                                 "model": args.model, "n_llm_candidates": 25}, False),
            }

            for cond_name, (fn, kwargs, is_warmstart) in conditions.items():
                t0 = time.time()
                print(f"    {cond_name}...", end="", flush=True)

                for seed_idx in range(args.n_seeds):
                    if is_warmstart:
                        disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                        result = run_ls_na_egbo_campaign(
                            disc_seed, budget=args.budget,
                            strategy_fn=fn, strategy_kwargs=kwargs,
                            batch_size=args.batch_size, seed=seed_idx,
                            n_init=args.n_init, n_propose=30,
                            protein=protein, prior_level=prior_level,
                            model=args.model, mock_llm=args.mock_llm,
                        )
                    else:
                        disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)
                        X_init, Y_init = shared_inits[seed_idx]
                        result = run_mo_campaign(
                            disc_seed, X_init, Y_init, args.budget,
                            fn, kwargs, batch_size=args.batch_size, seed=seed_idx,
                        )

                    final_hv = result["hv_trajectory"][-1] if result["hv_trajectory"] else float("nan")
                    final_pf = len(pareto_front_of(result["Y_obs"]))
                    igd = compute_igd(result["Y_obs"], true_pareto, ref_point)
                    exp_to_90 = experiments_to_threshold(
                        result["hv_trajectory"], args.n_init, args.batch_size,
                        0.9, max_hv)
                    stag = count_stagnant_batches(result["hv_trajectory"])

                    results.append({
                        "phase": 2, "protein": protein, "prior_level": prior_level,
                        "condition": cond_name, "seed": seed_idx,
                        "budget": args.budget, "n_init": args.n_init,
                        "final_hv": final_hv, "final_pf_size": final_pf,
                        "igd": igd, "exp_to_90pct_hv": exp_to_90,
                        "stagnant_batches": stag,
                        "max_hv": max_hv,
                        "n_obs": len(result["Y_obs"]),
                    })
                    print(".", end="", flush=True)
                elapsed = time.time() - t0
                print(f" ({elapsed:.0f}s)")

    df = pd.DataFrame(results)
    return df


# ── Statistical analysis ───────────────────────────────────────────────────────

def run_statistical_tests(df, baseline="mo_egbo_novelty"):
    """
    Run Wilcoxon signed-rank tests of each condition vs the baseline.
    Apply Holm-Bonferroni correction.
    """
    results = []

    # Group by phase, protein, prior_level
    for (phase, protein, prior), group in df.groupby(["phase", "protein", "prior_level"]):
        conditions = group["condition"].unique()
        baseline_data = group[group["condition"] == baseline]["final_hv"].values

        pvals = {}
        for cond in conditions:
            if cond == baseline:
                continue
            cond_data = group[group["condition"] == cond]["final_hv"].values
            # Align by seed
            baseline_seeds = group[group["condition"] == baseline]["seed"].values
            cond_seeds = group[group["condition"] == cond]["seed"].values
            common_seeds = sorted(set(baseline_seeds) & set(cond_seeds))
            b_vals = np.array([baseline_data[list(baseline_seeds).index(s)] for s in common_seeds])
            c_vals = np.array([cond_data[list(cond_seeds).index(s)] for s in common_seeds])

            if len(common_seeds) >= 5:
                try:
                    stat, pval = stats.wilcoxon(b_vals, c_vals, alternative="greater")
                except Exception:
                    pval = 1.0
            else:
                pval = 1.0
            pvals[cond] = pval

        # Holm-Bonferroni correction
        sorted_conds = sorted(pvals.keys(), key=lambda c: pvals[c])
        m = len(sorted_conds)
        for rank, cond in enumerate(sorted_conds):
            corrected_p = min(pvals[cond] * (m - rank), 1.0)
            results.append({
                "phase": phase, "protein": protein, "prior_level": prior,
                "baseline": baseline, "condition": cond,
                "raw_pval": pvals[cond],
                "holm_corrected_pval": corrected_p,
                "significant": corrected_p < 0.05,
                "baseline_mean_hv": float(np.nanmean(baseline_data)),
                "condition_mean_hv": float(
                    np.nanmean(group[group["condition"] == cond]["final_hv"].values)),
            })

    return pd.DataFrame(results)


# ── Summary table ──────────────────────────────────────────────────────────────

def summarize_results(df):
    """Produce a summary table with mean ± std per condition."""
    summary = df.groupby(["phase", "protein", "prior_level", "condition"]).agg(
        hv_mean=("final_hv", "mean"),
        hv_std=("final_hv", "std"),
        hv_median=("final_hv", "median"),
        pf_mean=("final_pf_size", "mean"),
        igd_mean=("igd", "mean"),
        igd_std=("igd", "std"),
        exp90_mean=("exp_to_90pct_hv", "mean"),
        exp90_std=("exp_to_90pct_hv", "std"),
        stag_mean=("stagnant_batches", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-augmented formulation optimization benchmark")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Which phase to run")
    parser.add_argument("--n_seeds", type=int, default=20,
                        help="Number of random seeds per condition")
    parser.add_argument("--budget", type=int, default=30,
                        help="Total experiment budget")
    parser.add_argument("--n_init", type=int, default=10,
                        help="Number of initial experiments")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Experiments per batch")
    parser.add_argument("--model", default="qwen3:32b",
                        help="Ollama model for LLM conditions")
    parser.add_argument("--mock_llm", action="store_true",
                        help="Use mock LLM (no Ollama needed)")
    parser.add_argument("--proteins", nargs="+",
                        default=["mAb_aggregation", "mAb_oxidation"],
                        help="Protein profiles to test")
    parser.add_argument("--priors", nargs="+",
                        default=["L1", "blank"],
                        help="Prior knowledge levels: L1, blank, wrong")
    parser.add_argument("--w_acq", type=float, default=0.9,
                        help="Acquisition weight for novelty selection")
    parser.add_argument("--w_nov", type=float, default=0.1,
                        help="Novelty weight for novelty selection")
    parser.add_argument("--out_dir", default="/mnt/results/benchmark",
                        help="Output directory")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run (default: all)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark configuration:")
    print(f"  Phase: {args.phase}")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Budget: {args.budget}, N_init: {args.n_init}, Batch: {args.batch_size}")
    print(f"  Model: {args.model}, Mock: {args.mock_llm}")
    print(f"  Proteins: {args.proteins}")
    print(f"  Priors: {args.priors}")
    print(f"  Novelty weights: w_acq={args.w_acq}, w_nov={args.w_nov}")

    t_start = time.time()

    if args.phase == 1:
        df = run_phase1(args)
    elif args.phase == 2:
        df = run_phase2(args)
    else:
        print("Phase 3 (Ada coatings) not yet implemented in this runner.")
        return

    elapsed = time.time() - t_start
    print(f"\n\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save raw results
    df.to_csv(out_dir / f"phase{args.phase}_raw_results.csv", index=False)
    print(f"Saved raw results: {out_dir / f'phase{args.phase}_raw_results.csv'}")

    # Summary
    summary = summarize_results(df)
    summary.to_csv(out_dir / f"phase{args.phase}_summary.csv", index=False)
    print(f"Saved summary: {out_dir / f'phase{args.phase}_summary.csv'}")

    # Statistical tests
    stats_df = run_statistical_tests(df, baseline="mo_egbo_novelty")
    stats_df.to_csv(out_dir / f"phase{args.phase}_stats_vs_egbo_novelty.csv", index=False)
    print(f"Saved stats: {out_dir / f'phase{args.phase}_stats_vs_egbo_novelty.csv'}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY (Phase {args.phase})")
    print(f"{'='*80}")
    for (protein, prior), group in summary.groupby(["protein", "prior_level"]):
        print(f"\n  Protein: {protein}, Prior: {prior}")
        print(f"  {'Condition':<20} {'HV mean±std':>15} {'PF size':>8} {'IGD':>8} {'Exp→90%':>8} {'Stag':>5}")
        print(f"  {'-'*75}")
        for _, row in group.iterrows():
            print(f"  {row['condition']:<20} {row['hv_mean']:>8.0f}±{row['hv_std']:<5.0f} "
                  f"{row['pf_mean']:>8.1f} {row['igd_mean']:>8.1f} "
                  f"{row['exp90_mean']:>8.1f} {row['stag_mean']:>5.1f}")

    # Print key statistical comparisons
    if len(stats_df) > 0:
        print(f"\n{'='*80}")
        print(f"STATISTICAL TESTS (vs mo_egbo_novelty baseline)")
        print(f"{'='*80}")
        sig = stats_df[stats_df["significant"]]
        if len(sig) > 0:
            print(f"\n  Significant differences (Holm-corrected p<0.05):")
            for _, row in sig.iterrows():
                direction = ">" if row["baseline_mean_hv"] > row["condition_mean_hv"] else "<"
                print(f"    {row['baseline']} {direction} {row['condition']} "
                      f"(p={row['holm_corrected_pval']:.4f}, "
                      f"{row['protein']}/{row['prior_level']})")
        else:
            print(f"\n  No significant differences found at p<0.05 after Holm correction.")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
