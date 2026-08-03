"""
ablation_batch_scoring.py — Decompose EGBO advantage into batch selection vs scoring quality.

2x2 design:
                  hand-coded scorer    LLM scorer
  joint batch=4   egbo_sklearn         egbo_llm_scorer      (LLM writes scoring only)
  sequential=1    egbo_batch1          llm_c_evo (existing)

Run on pareto_20210112, 20 shared seeds (same inits as previous ablations).

Usage:
    python ablation_batch_scoring.py \\
        --data_dir data/ \\
        --out_dir results_ablation/ \\
        --n_repeats 20 \\
        --datasets pareto_20210112 \\
        --model qwen2.5-coder:7b
"""

import argparse
import json
import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from oracle import NNOracle
from evaluate import compute_metrics, aggregate_across_seeds
from ablation_gp_calibration import (
    run_egbo_sklearn_gp, generate_shared_inits
)
from shared_seed_experiment import _save_and_print


def _robust_parse_scores(text, expected_n):
    """Extract float acquisition scores from LLM response, handling markdown/prose."""
    import re, json as _json
    import numpy as _np

    # Strip markdown fences
    clean = text.strip()
    for fence in ['```json', '```']:
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith('```'):
        clean = clean[:-3]
    clean = clean.strip()

    # Try 1: direct json.loads
    try:
        parsed = _json.loads(clean)
        if 'scores' in parsed:
            s = _np.array(parsed['scores'], dtype=float)
            if len(s) == expected_n and _np.all(_np.isfinite(s)):
                return s
    except Exception:
        pass

    # Try 2: extract "scores" array directly
    m = re.search(r'"scores"\s*:\s*(\[[\d\s.,eE+\-]+\])', text, re.DOTALL)
    if m:
        try:
            s = _np.array(_json.loads(m.group(1)), dtype=float)
            if len(s) == expected_n and _np.all(_np.isfinite(s)):
                return s
        except Exception:
            pass

    # Try 3: any JSON object with DOTALL
    for m in re.finditer(r'\{[^{}]+\}', text, re.DOTALL):
        try:
            parsed = _json.loads(m.group())
            if 'scores' in parsed:
                s = _np.array(parsed['scores'], dtype=float)
                if len(s) == expected_n and _np.all(_np.isfinite(s)):
                    return s
        except Exception:
            continue

    # Try 4: any array of the right length
    m = re.search(r'\[[\d\s.,eE+\-]+\]', text, re.DOTALL)
    if m:
        try:
            s = _np.array(_json.loads(m.group()), dtype=float)
            if len(s) == expected_n and _np.all(_np.isfinite(s)):
                return s
        except Exception:
            pass

    return None


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_DATASETS = [
    ("coatings",        "coatings"),
    ("pareto_20201218", "pareto_campaign 2020-12-18_17-38-40"),
    ("pareto_20201223", "pareto_campaign 2020-12-23_17-06-50"),
    ("pareto_20210104", "pareto_campaign 2021-01-04_08-37-39"),
    ("pareto_20210112", "pareto_campaign 2021-01-12_16-26-56"),
]


# ── EGBO with batch_size=1 (sequential selection) ────────────────────────────

def run_egbo_batch1(oracle, X_init, y_init, budget, merit_weight=1.0,
                    random_state=0, **kwargs):
    """
    EGBO with batch_size=1: selects one candidate per iteration.
    Everything else (sklearn GP, evolutionary candidates, hand-coded UCB scoring)
    is identical to egbo_sklearn. Tests whether joint batch selection is the mechanism.
    """
    return run_egbo_sklearn_gp(
        oracle, X_init, y_init, budget,
        batch_size=1,           # ← the only change
        qnehvi_candidates=8,
        evo_candidates=72,
        merit_weight=merit_weight,
        random_state=random_state,
    )


# ── EGBO with LLM-written acquisition scoring ────────────────────────────────

def run_egbo_llm_scorer(oracle, X_init, y_init, budget, model="qwen2.5-coder:7b",
                        batch_size=4, evo_candidates=72, merit_weight=1.0,
                        random_state=0, **kwargs):
    """
    EGBO pipeline but the acquisition scoring is written by the LLM.
    
    The standard EGBO loop generates candidates via U-NSGA-III and the GP.
    Instead of hand-coded UCB(mu + 2*sigma), we pass the candidate pool 
    (X_cands, mu, sigma, X_obs, y_obs, bounds) to the LLM and ask it to 
    output acquisition scores as a numpy array.
    
    This isolates scoring quality from batch selection:
    - If this matches egbo_sklearn: LLM scoring is equivalent to hand-coded
    - If this underperforms: LLM scoring is the bottleneck, not batch selection
    """
    import ollama
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import MinMaxScaler
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    warnings.filterwarnings("ignore")

    bounds_np = oracle.bounds()
    d = bounds_np.shape[0]
    N_dataset = len(oracle._X_raw)
    all_X = oracle._X_raw
    scaler_oracle = oracle._scaler

    def norm_x(X):
        lo, hi = bounds_np[:, 0], bounds_np[:, 1]
        return (X - lo) / (hi - lo + 1e-12)

    def denorm_x(X_n):
        lo, hi = bounds_np[:, 0], bounds_np[:, 1]
        return X_n * (hi - lo) + lo

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    X_scaled_all = scaler_oracle.transform(all_X)
    X_init_scaled = scaler_oracle.transform(X_init)
    queried = set()
    for row in X_init_scaled:
        dists = np.linalg.norm(X_scaled_all - row, axis=1)
        queried.add(int(np.argmin(dists)))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []
    n_batches = max(1, (budget - len(X_init)) // batch_size)

    _llm_scoring_prompt = """\
You are selecting an acquisition function for a Bayesian optimisation campaign.
Choose ONE and return JSON only. Python will compute the scores.

Options:
  {"acq": "ucb", "beta": <float>}  — UCB: high beta=explore, low=exploit
  {"acq": "ei",  "xi": <float>}    — Expected Improvement, xi=0.01 standard
  {"acq": "pi",  "xi": <float>}    — Probability of Improvement, conservative

Guidelines: early/uncertain → ucb beta=5-20; mid → ucb beta=1-3 or ei; late → ei or pi
Respond with JSON only. No explanation."""

    for batch_idx in range(n_batches):
        X_obs_norm = norm_x(X_obs)

        # Fit sklearn GP (identical to egbo_sklearn)
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
            normalize_y=True, n_restarts_optimizer=3, alpha=1e-6,
        )
        gp.fit(X_obs_norm, y_obs)

        # Generate EA candidates (identical to egbo_sklearn)
        top_k = min(evo_candidates, len(X_obs))
        top_idx_ea = np.argsort(y_obs)[-top_k:][::-1]
        seed_x = X_obs_norm[top_idx_ea]
        if seed_x.shape[0] < evo_candidates:
            rng_pad = np.random.default_rng(random_state + batch_idx)
            pad = rng_pad.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])

        try:
            ref_dirs = get_reference_directions("energy", 1, evo_candidates, seed=random_state)
        except Exception:
            ref_dirs = np.random.default_rng(random_state).random((evo_candidates, 1))

        try:
            algo = UNSGA3(pop_size=evo_candidates, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=1, n_constr=0, xl=np.zeros(d), xu=np.ones(d))
            algo.setup(pm, termination=NoTermination())
            pop = algo.ask()
            pop_size_actual = len(pop)
            f_vals = -y_obs[np.argsort(y_obs)[-pop_size_actual:]]
            pop.set("F", f_vals.reshape(-1, 1))
            algo.tell(infills=pop)
            ea_cands_norm = np.clip(algo.ask().get("X"), 0, 1)
        except Exception:
            ea_cands_norm = np.random.default_rng(random_state + batch_idx).random(
                (evo_candidates, d))

        rng_bo = np.random.default_rng(random_state + batch_idx + 1000)
        bo_cands_norm = rng_bo.random((8, d))

        all_cands_norm = np.vstack([bo_cands_norm, ea_cands_norm])
        mu, sigma = gp.predict(all_cands_norm, return_std=True)

        # ── LLM SCORING: choose acquisition function, Python computes scores ──
        progress = (len(X_obs) - len(X_init)) / max(budget - len(X_init), 1)
        gp_unc = float(sigma.mean())
        improvement_rate_val = sum(
            y_obs[-10:][i] > y_obs[-10:][:i].max()
            for i in range(1, min(10, len(y_obs)))
        ) / max(min(10, len(y_obs)) - 1, 1)
        user_prompt = (
            f"progress: {progress:.2f}\n"
            f"gp_uncertainty: {gp_unc:.4f}\n"
            f"improvement_rate: {improvement_rate_val:.2f}\n"
            f"best_normalised: {float(y_obs.max()/(oracle.global_best()+1e-12)):.3f}\n"
        )

        acq_scores = mu + 2.0 * sigma  # fallback: UCB beta=2
        acq_choice = "ucb_fallback"
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _llm_scoring_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={"temperature": 0.1, "num_predict": 64},
            )
            text = response["message"]["content"]
            import re as _re, json as _json
            m = _re.search(r'\{[^{}]*\}', text, _re.DOTALL)
            if m:
                parsed = _json.loads(m.group())
                acq = parsed.get("acq", "ucb").lower()
                if acq == "ucb":
                    beta = float(max(0.01, min(400.0, parsed.get("beta", 2.0))))
                    acq_scores = mu + beta * sigma
                    acq_choice = f"ucb(b={beta:.1f})"
                elif acq == "ei":
                    from scipy.stats import norm as _norm
                    xi = float(parsed.get("xi", 0.01))
                    y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
                    y_best_s = (float(y_obs.max()) - y_mean) / y_std
                    mu_s = (mu - y_mean) / y_std
                    sigma_s = np.maximum(sigma, 1e-9)
                    z = (mu_s - y_best_s - xi) / sigma_s
                    acq_scores = sigma_s * (_norm.cdf(z) * z + _norm.pdf(z))
                    acq_choice = f"ei(xi={xi})"
                elif acq == "pi":
                    from scipy.stats import norm as _norm
                    xi = float(parsed.get("xi", 0.01))
                    y_mean, y_std = y_obs.mean(), y_obs.std() + 1e-12
                    y_best_s = (float(y_obs.max()) - y_mean) / y_std
                    mu_s = (mu - y_mean) / y_std
                    acq_scores = _norm.cdf((mu_s - y_best_s - xi) / np.maximum(sigma, 1e-9))
                    acq_choice = f"pi(xi={xi})"
        except Exception as e:
            logger.warning(f"LLM scorer batch {batch_idx}: {e} — using UCB fallback")
        decisions.append((
            len(X_init) + batch_idx * batch_size, f"egbo_llm:{acq_choice}", {}
        ))
        # ─────────────────────────────────────────────────────────────────

        # Joint batch selection (same as standard EGBO, batch_size=4)
        selected_indices = []
        remaining = list(range(len(all_cands_norm)))
        for _ in range(batch_size):
            if not remaining:
                break
            rem = np.array(remaining)
            already = (all_cands_norm[selected_indices]
                       if selected_indices else np.zeros((0, d)))
            obs_combined = np.vstack([X_obs_norm, already]) if len(already) else X_obs_norm
            nov = np.array([
                np.min(np.linalg.norm(obs_combined - all_cands_norm[i], axis=1))
                for i in rem
            ])
            a = acq_scores[rem]
            a_lo, a_hi = a.min(), a.max()
            a_norm = (a - a_lo) / (a_hi - a_lo + 1e-12)
            n_lo, n_hi = nov.min(), nov.max()
            n_norm = (nov - n_lo) / (n_hi - n_lo + 1e-12)
            score = merit_weight * a_norm + (1 - merit_weight) * n_norm
            pick = int(rem[np.argmax(score)])
            selected_indices.append(pick)
            remaining.remove(pick)

        # Snap to nearest unqueried dataset rows (identical to all other EGBO variants)
        unqueried = [i for i in range(N_dataset) if i not in queried]
        if not unqueried:
            unqueried = list(range(N_dataset))

        new_x_rows, new_y_vals = [], []
        for idx in selected_indices:
            x_raw = denorm_x(all_cands_norm[idx])
            x_s = scaler_oracle.transform(x_raw.reshape(1, -1))[0]
            pool_s = scaler_oracle.transform(all_X[unqueried])
            dists = np.linalg.norm(pool_s - x_s, axis=1)
            chosen = unqueried[int(np.argmin(dists))]
            queried.add(chosen)
            new_x_rows.append(all_X[chosen])
            new_y_vals.append(float(oracle._y_raw[chosen]))
            unqueried = [i for i in unqueried if i != chosen]

        X_obs = np.vstack([X_obs, np.array(new_x_rows)])
        y_obs = np.append(y_obs, new_y_vals)
        for _ in new_y_vals:
            running_best.append(float(y_obs.max()))

        decisions.append((
            len(X_init) + batch_idx * batch_size, "egbo_llm_scorer", {}
        ))

    return {
        "running_best": running_best, "decisions": decisions,
        "failures": [], "X_obs": X_obs, "y_obs": y_obs,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--out_dir",     default="results_ablation")
    parser.add_argument("--n_repeats",   type=int, default=20)
    parser.add_argument("--n_init",      type=int, default=10)
    parser.add_argument("--budget_frac", type=float, default=0.5)
    parser.add_argument("--datasets",    nargs="+", default=None)
    parser.add_argument("--model",       default="qwen2.5-coder:7b")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_list = ALL_DATASETS
    if args.datasets:
        dataset_list = [(l, n) for l, n in ALL_DATASETS if l in args.datasets]

    all_rows = []

    for ds_label, ds_name in dataset_list:
        print(f"\n{'='*60}\nDataset: {ds_label}")
        oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        N      = len(oracle._X_raw)
        budget = max(args.n_init + 10, int(args.budget_frac * N))
        gb     = oracle.global_best()
        gmin   = float(oracle._y_raw.min())
        print(f"  N={N}  budget={budget}  dims={oracle.bounds().shape[0]}")

        # Use same seed=42 shared inits as all previous ablations for comparability
        shared_inits = generate_shared_inits(oracle, args.n_repeats,
                                              args.n_init, rng_seed=42)
        best_norms = [y.max()/gb for _, y in shared_inits]
        print(f"  Shared inits: best_norm {min(best_norms):.2f}–{max(best_norms):.2f}")

        ablations = [
            # label, runner, extra_kwargs
            ("egbo_batch1",     run_egbo_batch1,      {"merit_weight": 1.0}),
            ("egbo_llm_scorer", run_egbo_llm_scorer,  {"merit_weight": 1.0,
                                                        "model": args.model}),
        ]

        for label, runner, extra in ablations:
            seed_metrics, seed_curves, seed_logs = [], [], []
            for rep_idx, (X_init, y_init) in enumerate(shared_inits):
                try:
                    res = runner(
                        oracle, X_init, y_init, budget,
                        batch_size=4, evo_candidates=72,
                        random_state=rep_idx, **extra,
                    )
                    m = compute_metrics(res, gb, budget=budget, oracle_global_min=gmin)
                    seed_metrics.append(m)
                    seed_curves.append(np.array(res["running_best"]))
                    seed_logs.append({"decisions": res["decisions"], "failures": []})
                except Exception as e:
                    logger.warning(f"{label} rep {rep_idx} failed: {e}")

            _save_and_print(out_dir, ds_label, label, seed_metrics,
                            seed_curves, seed_logs, gb, gmin, shared_inits, all_rows)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_dir / "ablation_batch_scoring.csv", index=False)
    print(f"\nSaved ablation_batch_scoring.csv ({len(out_df)} rows)")

    # Print 2x2 table with existing results
    print(f"\n{'='*60}")
    print("2x2 DECOMPOSITION: batch selection × scoring quality")
    print(f"{'='*60}")
    print()

    # Load existing egbo_sklearn result for comparison
    existing_path = out_dir / "ablation_summary.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
    else:
        existing = pd.DataFrame()

    # Reference values from 80-seed power run (conditioned mid-AUC)
    reference = {
        "egbo_sklearn (joint, hand-coded)":    ("joint=4", "hand-coded",  0.642, 0.108),
        "egbo_batch1  (seq=1, hand-coded)":    ("seq=1",   "hand-coded",  None,  None),
        "egbo_llm_scorer (joint, LLM)":        ("joint=4", "LLM",         None,  None),
        "llm_c_evo   (seq=1, LLM)":            ("seq=1",   "LLM",         0.577, 0.111),
    }

    print(f"  {'Condition':<40} {'Batch':>8} {'Scorer':>12} {'mid-AUC':>9}")
    print(f"  {'-'*72}")

    for ds in out_df.dataset.unique():
        print(f"\n  {ds}:")
        sub = out_df[out_df.dataset == ds]

        rows_to_show = [
            ("egbo_sklearn",   "joint=4", "hand-coded", 0.642),  # from power run
            ("egbo_batch1",    "seq=1",   "hand-coded", None),
            ("egbo_llm_scorer","joint=4", "LLM",        None),
            ("llm_c_evo",      "seq=1",   "LLM",        0.577),  # from power run
        ]
        for cond, batch, scorer, ref_auc in rows_to_show:
            row = sub[sub.condition == cond]
            if len(row):
                m = row["auc_best"].mean()
                s = row["auc_best"].std()
                print(f"    {cond:<30} {batch:>8} {scorer:>12} {m:>9.3f}±{s:.3f}")
            elif ref_auc is not None:
                print(f"    {cond:<30} {batch:>8} {scorer:>12} {ref_auc:>9.3f} (from power run)")
            else:
                print(f"    {cond:<30} {batch:>8} {scorer:>12} {'—':>9}")

    print()
    print("Interpretation guide:")
    print("  If egbo_batch1 ≈ llm_c_evo: batch selection is the mechanism")
    print("  If egbo_llm_scorer ≈ egbo_sklearn: LLM scoring is adequate")
    print("  If egbo_llm_scorer ≈ llm_c_evo: scoring quality is the mechanism")
    print("  If both gaps survive: both contribute")


if __name__ == "__main__":
    main()
