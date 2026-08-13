"""
run_benchmark_resumable.py — Resumable Phase 1/2 benchmark with incremental checkpointing.

Writes results to a checkpoint CSV after every single seed, so progress survives
sandbox hibernation. On restart, loads the checkpoint and skips already-completed
(protein, prior, condition, seed) combos.

Usage:
    python run_benchmark_resumable.py --phase 1 --mock_llm --n_seeds 15 --budget 30 \
        --out_dir /mnt/results/benchmark_phase1

    # Restart after hibernation — just re-run the same command, it resumes.
"""

import argparse
import json
import pathlib
import sys
import threading
import time
import warnings
import concurrent.futures
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
    strategy_mo_llm, strategy_mo_scalarized_ucb, run_mo_campaign,
    make_shared_inits, pareto_front_of, OBJECTIVE_DIRECTIONS, OBJECTIVE_NAMES,
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
    "igd", "exp_to_90pct_hv", "exp_to_70pct_hv", "stagnant_batches", "max_hv", "n_obs",
    "llm_fallback_used", "llm_retry_count",
]


def load_checkpoint(ckpt_path):
    """Load existing checkpoint CSV, return DataFrame or empty."""
    if pathlib.Path(ckpt_path).exists():
        try:
            df = pd.read_csv(ckpt_path)
            print(f"  Loaded checkpoint: {len(df)} rows from {ckpt_path}")
            return df
        except Exception as e:
            print(f"  Warning: could not load checkpoint: {e}")
    return pd.DataFrame(columns=CKPT_COLS)


def save_checkpoint(df, ckpt_path):
    """Save checkpoint CSV. S3-backed /mnt/results doesn't support os.replace,
    so write directly (CSV is append-safe format)."""
    df.to_csv(str(ckpt_path), index=False)


def is_completed(ckpt_df, phase, protein, prior, condition, seed):
    """Check if a specific combo is already in the checkpoint."""
    if len(ckpt_df) == 0:
        return False
    mask = (
        (ckpt_df["phase"] == phase) &
        (ckpt_df["protein"] == protein) &
        (ckpt_df["prior_level"] == prior) &
        (ckpt_df["condition"] == condition) &
        (ckpt_df["seed"] == seed)
    )
    return mask.any()


# ── Phase runner (shared by Phase 1 and 2) ─────────────────────────────────────

# Serialises oracle.make_discrete_oracle() calls (shared, stateful oracle.rng)
# and checkpoint writes across concurrent seed threads within a single
# run_phase() call. Module-level is fine since run_phase runs single-threaded
# at the top level; only the seed-level ThreadPoolExecutor inside it is
# concurrent.
_ORACLE_RNG_LOCK = threading.Lock()
_RESULTS_LOCK = threading.Lock()


def run_phase(args, phase_num, n_propose, n_llm_cands):
    """
    Unified phase runner with incremental checkpointing.
    phase_num: 1 or 2
    n_propose: number of EGBO proposals per batch (25 for P1, 30 for P2)
    n_llm_cands: LLM candidates for LABO (20 for P1, 25 for P2)
    """
    print(f"\n{'='*60}")
    print(f"PHASE {phase_num}: {'Reduced' if phase_num == 1 else 'Full 14-excipient'} space")
    print(f"{'='*60}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"phase{phase_num}_raw_results.csv"

    ckpt_df = load_checkpoint(ckpt_path)
    results = ckpt_df.to_dict("records")

    n_conditions = len(args.conditions) if args.conditions else 8
    total_combos = len(args.proteins) * len(args.priors) * n_conditions * args.n_seeds
    done_combos = len(results)
    print(f"  Progress: {done_combos}/{total_combos} combos completed")

    for protein in args.proteins:
        for prior_level in args.priors:
            print(f"\n  Protein: {protein}, Prior: {prior_level}")

            oracle = MultiObjectiveExcipientOracle(
                protein=protein,
                tm_noise=0.013, kd_noise=0.096, viscosity_noise=0.10,
                seed=42,
            )
            disc = oracle.make_discrete_oracle(n_samples=500, seed=42)

            # Ground truth Pareto front for IGD
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
                "mo_ls_egbo": (strategy_mo_egbo_real, {}, True),  # LLM warm-start + real EGBO, no novelty
                "mo_ls_na_egbo": (strategy_mo_ls_na_egbo,
                                  {"w_acq": args.w_acq, "w_nov": args.w_nov}, True),
                # LLM warm-start + scalarised UCB (simpler acquisition than
                # the full qLogNEHVI/novelty backbone) -- does warm-start
                # help independent of acquisition strength? Same ablation
                # added to run_phase3.py's coatings conditions; seeds here
                # are already comparable across conditions since every
                # condition in this dict runs seed_idx 0..n_seeds-1 through
                # either the same shared_inits or the same warm-start seed.
                "mo_ucb_warmstart": (strategy_mo_scalarized_ucb,
                                      {"beta": 2.0}, True),
                "mo_llm_candidate_gen": (strategy_mo_llm_candidate_gen,
                                {"prior_text": prior_text, "mock_llm": args.mock_llm,
                                 "model": args.model, "n_llm_candidates": n_llm_cands,
                                 "n_init": args.n_init, "budget": args.budget,
                                 "llm_taper_frac": args.llm_taper_frac,
                                 "w_acq": args.w_acq, "w_nov": args.w_nov}, False),
            }

            if args.conditions:
                unknown = set(args.conditions) - set(conditions)
                if unknown:
                    raise ValueError(f"Unknown condition(s) in --conditions: {sorted(unknown)}. "
                                      f"Valid choices: {sorted(conditions)}")
                conditions = {k: v for k, v in conditions.items() if k in args.conditions}

            for cond_name, (fn, kwargs, is_warmstart) in conditions.items():
                t0 = time.time()
                pending = [s for s in range(args.n_seeds)
                           if not is_completed(ckpt_df, phase_num, protein, prior_level, cond_name, s)]
                skipped = args.n_seeds - len(pending)
                ran = 0

                def run_one_seed(seed_idx):
                    # oracle.make_discrete_oracle() draws noise from the shared,
                    # stateful oracle.rng — not safe for concurrent threads to
                    # call at once, so it's serialised. This part is cheap
                    # relative to the LLM call / EGBO stage that follows, so
                    # serialising it doesn't meaningfully reduce parallelism.
                    with _ORACLE_RNG_LOCK:
                        disc_seed = oracle.make_discrete_oracle(n_samples=500, seed=42)

                    if is_warmstart:
                        result = run_ls_na_egbo_campaign(
                            disc_seed, budget=args.budget,
                            strategy_fn=fn, strategy_kwargs=kwargs,
                            batch_size=args.batch_size, seed=seed_idx,
                            n_init=args.n_init, n_propose=n_propose,
                            protein=protein, prior_level=prior_level,
                            model=args.model, mock_llm=args.mock_llm,
                        )
                    else:
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
                    exp_to_70 = experiments_to_threshold(
                        result["hv_trajectory"], args.n_init, args.batch_size,
                        0.7, max_hv)
                    stag = count_stagnant_batches(result["hv_trajectory"])

                    return {
                        "phase": phase_num, "protein": protein, "prior_level": prior_level,
                        "condition": cond_name, "seed": seed_idx,
                        "budget": args.budget, "n_init": args.n_init,
                        "final_hv": final_hv, "final_pf_size": final_pf,
                        "igd": igd, "exp_to_90pct_hv": exp_to_90,
                        "exp_to_70pct_hv": exp_to_70,
                        "stagnant_batches": stag,
                        "max_hv": max_hv,
                        "n_obs": len(result["Y_obs"]),
                        "llm_fallback_used": result.get("llm_fallback_used", False),
                        "llm_retry_count": result.get("llm_retry_count", 0),
                    }

                with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as ex:
                    futures = {ex.submit(run_one_seed, s): s for s in pending}
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        with _RESULTS_LOCK:
                            results.append(row)
                            ran += 1
                            # Incremental checkpoint after every seed
                            df_tmp = pd.DataFrame(results, columns=CKPT_COLS)
                            save_checkpoint(df_tmp, ckpt_path)

                elapsed = time.time() - t0
                status = f"ran {ran}" if ran > 0 else "all cached"
                if skipped > 0:
                    status += f", skipped {skipped}"
                print(f"    {cond_name}: {status} ({elapsed:.0f}s)")

    df = pd.DataFrame(results, columns=CKPT_COLS)
    return df


# ── Statistical analysis ───────────────────────────────────────────────────────

def run_statistical_tests(df, baseline="mo_egbo_novelty"):
    results = []
    for (phase, protein, prior), group in df.groupby(["phase", "protein", "prior_level"]):
        conditions = group["condition"].unique()
        baseline_data = group[group["condition"] == baseline]["final_hv"].values

        pvals = {}
        for cond in conditions:
            if cond == baseline:
                continue
            cond_data = group[group["condition"] == cond]["final_hv"].values
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


def summarize_results(df):
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
    parser = argparse.ArgumentParser(description="Resumable LLM-augmented formulation benchmark")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--n_seeds", type=int, default=15)
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Concurrent seed workers (threads) per condition. "
                             "Default 4 matches Ollama's OLLAMA_NUM_PARALLEL; "
                             "measured ~1.6x wall-clock speedup at 2-way "
                             "concurrency on this hardware for real-LLM calls.")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--proteins", nargs="+", default=["mAb_aggregation", "mAb_oxidation"])
    parser.add_argument("--priors", nargs="+", default=["L1", "blank"])
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run (default: all 8). Choices: "
                             "mo_random, mo_egbo, mo_egbo_real, mo_egbo_novelty, "
                             "mo_ls_egbo, mo_ls_na_egbo, mo_ucb_warmstart, "
                             "mo_llm_candidate_gen")
    parser.add_argument("--w_acq", type=float, default=0.9)
    parser.add_argument("--w_nov", type=float, default=0.1,
                        help="Novelty weight (0.1 = sweet spot from sensitivity sweep)")
    parser.add_argument("--llm_taper_frac", type=float, default=0.5,
                        help="mo_llm_candidate_gen only: fraction of campaign batches "
                             "that call the LLM before tapering to EGBO-only (e.g. 0.5 "
                             "= first half of batches). Has no effect on other conditions.")
    parser.add_argument("--out_dir", default="/mnt/results/benchmark_phase1")
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
    print(f"  LLM taper fraction (mo_llm_candidate_gen only): {args.llm_taper_frac}")

    t_start = time.time()

    if args.phase == 1:
        df = run_phase(args, phase_num=1, n_propose=25, n_llm_cands=20)
    elif args.phase == 2:
        df = run_phase(args, phase_num=2, n_propose=30, n_llm_cands=25)
    else:
        print("Phase 3 handled by run_phase3.py")
        return

    elapsed = time.time() - t_start
    print(f"\n\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save final results (checkpoint already has raw results)
    ckpt_path = out_dir / f"phase{args.phase}_raw_results.csv"
    df.to_csv(ckpt_path, index=False)
    print(f"Saved raw results: {ckpt_path}")

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

    if len(stats_df) > 0:
        print(f"\n{'='*80}")
        print(f"STATISTICAL TESTS (vs mo_egbo_novelty baseline)")
        print(f"{'='*80}")
        for _, row in stats_df.iterrows():
            sig = "***" if row["significant"] else ""
            print(f"  {row['protein']}/{row['prior_level']}: {row['condition']:<20} "
                  f"p={row['raw_pval']:.4f} (Holm {row['holm_corrected_pval']:.4f}) {sig}")


if __name__ == "__main__":
    main()
