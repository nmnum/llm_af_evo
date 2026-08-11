"""
run_phase3.py — Phase 3: Cross-domain generalisability on synthetic coatings.

Tests whether the LS-NA-EGBO architecture generalises to a different domain
(coatings) with only prompt changes — no code changes to the strategy itself.

The coatings problem is single-objective (maximise quality score), 4D continuous.
We adapt the multi-objective strategies to single-objective by using the same
acquisition (qLogNEHVI doesn't apply to single-objective, so we use qNEI instead)
and the same novelty-aware selection.

Conditions:
  random       — Random search
  egbo         — EGBO (GP + UCB + evolutionary candidates)
  egbo_novelty — EGBO + novelty-aware selection
  ls_na_egbo   — LLM warm-start + novelty-aware EGBO
  llm_labo     — LABO-style (LLM every batch)

Usage:
    python run_phase3.py --mock_llm --n_seeds 20
    python run_phase3.py --model qwen2.5:72b-instruct --n_seeds 20

    # Restart after interruption — just re-run the same command, it resumes
    # from results/benchmark_phase3/phase3_raw_results.csv (or --out_dir).

Resumable: writes a checkpoint row to <out_dir>/phase3_raw_results.csv
after every single (condition, seed) combo, not just at the end — the
real-LLM run (llm_labo calls the LLM every batch, ls_na_egbo once per
campaign) is long enough (~20hrs at qwen3:32b's measured ~327s/call,
~220 total calls across 20 seeds) that losing all progress to a crash or
disconnect partway through would be expensive to re-pay. On restart, any
(condition, seed) combo already present in the checkpoint is skipped.
"""

import argparse
import pathlib
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from synthetic_coatings_oracle import SyntheticCoatingsOracle
from novelty_selection import novelty_aware_select_vectorised
from llm_warmstart import llm_warmstart_init_coatings, build_coatings_prompt, parse_llm_coatings


CKPT_COLS = [
    "phase", "condition", "seed", "budget", "n_init", "final_best",
    "best_frac", "exp_to_90pct", "n_obs", "llm_fallback_used", "llm_retry_count",
]


def load_checkpoint(ckpt_path):
    """Load existing checkpoint CSV, return DataFrame or empty. Same pattern
    as run_benchmark_resumable.py's load_checkpoint."""
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


def is_completed(ckpt_df, condition, seed):
    if len(ckpt_df) == 0:
        return False
    mask = (ckpt_df["condition"] == condition) & (ckpt_df["seed"] == seed)
    return mask.any()


def make_shared_inits_coatings(oracle, n_repeats, n_init, rng_seed=42):
    """Generate shared random initialisations for coatings oracle."""
    rng = np.random.default_rng(rng_seed)
    inits = []
    for _ in range(n_repeats):
        idx = rng.choice(len(oracle._X_raw), n_init, replace=False)
        inits.append((oracle._X_raw[idx].copy(), oracle._y_raw[idx].copy()))
    return inits


# ── Single-objective strategies for coatings ──────────────────────────────────

def strategy_coatings_random(oracle, X_obs, y_obs, bounds, batch_size, rng, **kw):
    d = bounds.shape[0]
    return rng.uniform(0, 1, (batch_size, d)), {}


def strategy_coatings_egbo(oracle, X_obs, y_obs, bounds, batch_size, rng, **kw):
    """EGBO for single-objective: GP + UCB + evolutionary candidates."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)

    # Fit GP
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs, y_obs)

    # UCB acquisition
    progress = kw.get("progress", 0.5)
    beta = max(0.5, 5.0 * (1.0 - progress))

    # Evolutionary candidates (simple mutation around best points)
    top_k = min(20, len(y_obs))
    seed_x = Xn[np.argsort(y_obs)[-top_k:][::-1]]
    if len(seed_x) < 20:
        pad = rng.random((20 - len(seed_x), d))
        seed_x = np.vstack([seed_x, pad])

    # Perturbation-based evolutionary candidates
    evo_cands = []
    for base in seed_x[:20]:
        for _ in range(3):
            perturbed = np.clip(base + rng.normal(0, 0.1, d), 0, 1)
            evo_cands.append(perturbed)
    evo_cands = np.array(evo_cands[:30])

    # Random exploration candidates
    rand_cands = rng.random((10, d))

    all_cands_n = np.vstack([evo_cands, rand_cands])
    all_cands = all_cands_n * (hi - lo) + lo

    # Score with UCB
    mu, sigma = gp.predict(scaler.transform(all_cands), return_std=True)
    ucb = mu + beta * sigma

    # Greedy top-k selection
    top_idx = np.argsort(ucb)[-batch_size:]
    return all_cands[top_idx], {}


def strategy_coatings_egbo_novelty(oracle, X_obs, y_obs, bounds, batch_size, rng,
                                    w_acq=0.9, w_nov=0.1, **kw):
    """EGBO + novelty-aware selection for single-objective coatings."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs, y_obs)

    progress = kw.get("progress", 0.5)
    beta = max(0.5, 5.0 * (1.0 - progress))

    # Evolutionary candidates
    top_k = min(20, len(y_obs))
    seed_x = Xn[np.argsort(y_obs)[-top_k:][::-1]]
    if len(seed_x) < 20:
        pad = rng.random((20 - len(seed_x), d))
        seed_x = np.vstack([seed_x, pad])

    evo_cands = []
    for base in seed_x[:20]:
        for _ in range(3):
            perturbed = np.clip(base + rng.normal(0, 0.1, d), 0, 1)
            evo_cands.append(perturbed)
    evo_cands = np.array(evo_cands[:30])

    rand_cands = rng.random((10, d))
    all_cands_n = np.vstack([evo_cands, rand_cands])

    # Score with UCB
    mu, sigma = gp.predict(scaler.transform(all_cands_n * (hi - lo) + lo), return_std=True)
    ucb = mu + beta * sigma

    # Novelty-aware selection
    selected_idx = novelty_aware_select_vectorised(
        all_cands_n, ucb, batch_size, w_acq=w_acq, w_nov=w_nov, X_obs_n=Xn,
    )

    return all_cands_n[selected_idx] * (hi - lo) + lo, {"novelty_select": True}


def strategy_coatings_ls_na_egbo(oracle, X_obs, y_obs, bounds, batch_size, rng,
                                  w_acq=0.9, w_nov=0.1, **kw):
    """LS-NA-EGBO for coatings: same as egbo_novelty during campaign.
    The LLM warm-start happens before the campaign loop."""
    return strategy_coatings_egbo_novelty(
        oracle, X_obs, y_obs, bounds, batch_size, rng, w_acq=w_acq, w_nov=w_nov, **kw
    )


def strategy_coatings_llm_labo(oracle, X_obs, y_obs, bounds, batch_size, rng,
                                model="qwen3:32b", mock_llm=False,
                                w_acq=0.9, w_nov=0.1, n_llm_candidates=20, **kw):
    """LABO-style for coatings: LLM generates candidates every batch."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    Xn = (X_obs - lo) / (hi - lo + 1e-12)

    # Generate LLM candidates
    if mock_llm:
        rng_mock = np.random.default_rng(rng.integers(1e6))
        llm_cands_n = np.zeros((n_llm_candidates, d))
        for j in range(d):
            perm = rng_mock.permutation(n_llm_candidates)
            llm_cands_n[:, j] = (perm + rng_mock.uniform(0, 1, n_llm_candidates)) / n_llm_candidates
    else:
        prompt = build_coatings_prompt(n_llm_candidates)
        import ollama, json, re
        for attempt in range(3):
            try:
                resp = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert optimisation assistant. "
                                    "Respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.4, "num_predict": 1024, "think": False,
                              "seed": int(rng.integers(1_000_000)) + attempt},
                )
                text = resp["message"]["content"]
                llm_cands_n = parse_llm_coatings(text, n_llm_candidates, d)
                if len(llm_cands_n) > 0:
                    break
            except Exception:
                if attempt == 2:
                    llm_cands_n = np.random.uniform(0, 1, (n_llm_candidates, d))

    # EGBO candidates
    top_k = min(20, len(y_obs))
    seed_x = Xn[np.argsort(y_obs)[-top_k:][::-1]]
    if len(seed_x) < 20:
        pad = rng.random((20 - len(seed_x), d))
        seed_x = np.vstack([seed_x, pad])

    evo_cands = []
    for base in seed_x[:15]:
        for _ in range(2):
            perturbed = np.clip(base + rng.normal(0, 0.1, d), 0, 1)
            evo_cands.append(perturbed)
    evo_cands = np.array(evo_cands[:20])
    rand_cands = rng.random((5, d))

    all_cands_n = np.vstack([evo_cands, rand_cands, llm_cands_n])

    # Score with GP + UCB
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_obs)
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
        alpha=1e-6, normalize_y=True, n_restarts_optimizer=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xs, y_obs)

    progress = kw.get("progress", 0.5)
    beta = max(0.5, 5.0 * (1.0 - progress))
    mu, sigma = gp.predict(scaler.transform(all_cands_n * (hi - lo) + lo), return_std=True)
    ucb = mu + beta * sigma

    selected_idx = novelty_aware_select_vectorised(
        all_cands_n, ucb, batch_size, w_acq=w_acq, w_nov=w_nov, X_obs_n=Xn,
    )

    return all_cands_n[selected_idx] * (hi - lo) + lo, {
        "labo": True, "n_llm": len(llm_cands_n), "n_total": len(all_cands_n)
    }


# ── Campaign runner for coatings ───────────────────────────────────────────────

def run_coatings_campaign(oracle, X_init, y_init, budget, strategy_fn, strategy_kwargs,
                           batch_size=5, seed=0):
    """Run a single-objective coatings campaign."""
    bounds = oracle.bounds()
    rng = np.random.default_rng(seed)
    X_obs, y_obs = X_init.copy(), y_init.copy()

    # Reset queried and register init points
    oracle._queried = set()
    X_all_s = oracle._scaler.transform(oracle._X_raw)
    for row in oracle._scaler.transform(X_init):
        idx = int(np.argmin(np.linalg.norm(X_all_s - row, axis=1)))
        oracle._queried.add(idx)

    running_best = [float(y_obs.max())] * len(y_obs)
    n_batches = max(1, (budget - len(y_obs)) // batch_size)

    for b in range(n_batches):
        step = len(y_obs)
        progress = step / budget

        try:
            candidates, extra = strategy_fn(
                oracle=oracle, X_obs=X_obs, y_obs=y_obs, bounds=bounds,
                batch_size=batch_size, rng=rng, progress=progress,
                **strategy_kwargs,
            )
        except Exception as e:
            warnings.warn(f"Strategy failed at batch {b}: {e}")
            candidates = rng.uniform(0, 1, (batch_size, bounds.shape[0]))

        # Snap to nearest unqueried oracle point
        unqueried = [i for i in range(len(oracle._X_raw)) if i not in oracle._queried]
        if not unqueried:
            unqueried = list(range(len(oracle._X_raw)))

        for cand in candidates[:batch_size]:
            if not unqueried:
                break
            cand_s = oracle._scaler.transform(cand.reshape(1, -1))[0]
            pool_s = oracle._scaler.transform(oracle._X_raw[unqueried])
            chosen = unqueried[int(np.argmin(np.linalg.norm(pool_s - cand_s, axis=1)))]
            oracle._queried.add(chosen)
            X_obs = np.vstack([X_obs, oracle._X_raw[chosen]])
            y_obs = np.append(y_obs, float(oracle._y_raw[chosen]))
            running_best.append(float(y_obs.max()))
            unqueried = [i for i in unqueried if i != chosen]

    return {
        "running_best": running_best,
        "X_obs": X_obs,
        "y_obs": y_obs,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Coatings generalisability benchmark")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--model", default="qwen3:32b")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--w_acq", type=float, default=0.9)
    parser.add_argument("--w_nov", type=float, default=0.1)
    # Was hardcoded to /mnt/results/benchmark_phase3 (a path from a
    # different environment — not writable, not even present, here).
    # Relative to the repo root, alongside the other results/benchmark_*
    # directories this project already uses.
    parser.add_argument("--out_dir", default="results/benchmark_phase3")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "phase3_raw_results.csv"

    print(f"\n{'='*60}")
    print(f"PHASE 3: Coatings generalisability (synthetic 4D)")
    print(f"{'='*60}")
    print(f"  Seeds: {args.n_seeds}, Budget: {args.budget}")
    print(f"  N_init: {args.n_init}, Batch: {args.batch_size}")
    print(f"  Mock LLM: {args.mock_llm}")

    oracle = SyntheticCoatingsOracle(seed=42, noise_level=0.05)
    disc = oracle.make_discrete_oracle(n_samples=200, seed=42)
    global_best = disc.global_best()
    print(f"  Pool: {len(disc._X_raw)}, Global best: {global_best:.4f}")

    shared_inits = make_shared_inits_coatings(disc, args.n_seeds, args.n_init, rng_seed=42)

    conditions = {
        "random": (strategy_coatings_random, {}, False),
        "egbo": (strategy_coatings_egbo, {}, False),
        "egbo_novelty": (strategy_coatings_egbo_novelty, {}, False),
        "ls_na_egbo": (strategy_coatings_ls_na_egbo,
                        {"w_acq": args.w_acq, "w_nov": args.w_nov}, True),
        "llm_labo": (strategy_coatings_llm_labo,
                      {"mock_llm": args.mock_llm, "model": args.model,
                       "n_llm_candidates": 20}, False),
    }

    ckpt_df = load_checkpoint(ckpt_path)
    results = ckpt_df.to_dict("records")
    total_combos = len(conditions) * args.n_seeds
    print(f"  Progress: {len(results)}/{total_combos} combos completed")

    t_start = time.time()

    for cond_name, (fn, kwargs, is_warmstart) in conditions.items():
        t0 = time.time()
        pending = [s for s in range(args.n_seeds)
                   if not is_completed(ckpt_df, cond_name, s)]
        skipped = args.n_seeds - len(pending)
        print(f"\n  {cond_name}... ({skipped} already done, {len(pending)} to run)",
              end="", flush=True)

        for seed_idx in pending:
            disc_seed = oracle.make_discrete_oracle(n_samples=200, seed=42)

            llm_fallback_used, llm_retry_count = False, 0
            if is_warmstart:
                # LLM warm-start for coatings
                X_init, y_init, warmstart_meta = llm_warmstart_init_coatings(
                    disc_seed, n_propose=25, n_select=args.n_init,
                    model=args.model, mock=args.mock_llm, seed=seed_idx,
                )
                llm_fallback_used = warmstart_meta["llm_fallback_used"]
                llm_retry_count = warmstart_meta["llm_retry_count"]
            else:
                X_init, y_init = shared_inits[seed_idx]

            result = run_coatings_campaign(
                disc_seed, X_init, y_init, args.budget, fn, kwargs,
                batch_size=args.batch_size, seed=seed_idx,
            )

            final_best = float(result["y_obs"].max())
            best_frac = final_best / global_best

            # Experiments to 90% of global best
            exp_to_90 = args.budget
            for i, val in enumerate(result["running_best"]):
                if val >= 0.9 * global_best:
                    exp_to_90 = i + 1
                    break

            results.append({
                "phase": 3, "condition": cond_name, "seed": seed_idx,
                "budget": args.budget, "n_init": args.n_init,
                "final_best": final_best, "best_frac": best_frac,
                "exp_to_90pct": exp_to_90,
                "n_obs": len(result["y_obs"]),
                "llm_fallback_used": llm_fallback_used,
                "llm_retry_count": llm_retry_count,
            })
            # Checkpoint after every single (condition, seed) combo, not just
            # at the end — real-LLM runs here are long enough (~20hrs full
            # scale) that losing all progress to a crash/disconnect would be
            # expensive. ckpt_df is intentionally NOT refreshed from this
            # save within the loop (matches run_benchmark_resumable.py's
            # pattern) -- is_completed() is only consulted once per
            # condition above, so this is safe within a single process run;
            # a concurrent second process against the same out_dir is not
            # supported.
            save_checkpoint(pd.DataFrame(results), ckpt_path)
            print(".", end="", flush=True)
        print(f" ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\n\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    df = pd.DataFrame(results)

    # Summary
    summary = df.groupby("condition").agg(
        best_mean=("final_best", "mean"),
        best_std=("final_best", "std"),
        frac_mean=("best_frac", "mean"),
        frac_std=("best_frac", "std"),
        exp90_mean=("exp_to_90pct", "mean"),
        exp90_std=("exp_to_90pct", "std"),
    ).sort_values("best_mean", ascending=False)
    summary.to_csv(out_dir / "phase3_summary.csv")

    print(f"\n{'='*60}")
    print(f"PHASE 3 RESULTS")
    print(f"{'='*60}")
    print(f"  {'Condition':<20} {'Best mean±std':>15} {'Frac of GB':>12} {'Exp→90%':>10}")
    print(f"  {'-'*60}")
    for cond, row in summary.iterrows():
        print(f"  {cond:<20} {row['best_mean']:>8.4f}±{row['best_std']:<5.4f} "
              f"{row['frac_mean']:>8.1%}±{row['frac_std']:<4.1%} "
              f"{row['exp90_mean']:>8.1f}±{row['exp90_std']:<4.1f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
