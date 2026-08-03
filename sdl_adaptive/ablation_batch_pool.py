"""
ablation_batch_pool.py — Comprehensive batch size and pool composition analysis.

Study 1: Batch size
  batch_size ∈ {1, 2, 4, 8}
  re_fit_gp  ∈ {True, False}
  Fixed: pool=80 (72 EA + 8 BO), w=1.0 (no novelty), acq=UCB β=2

Study 2: Pool composition (fractional factorial first pass)
  ea_ratio   ∈ {0.0, 0.2, 0.5, 0.9, 1.0}
  pool_size  ∈ {20, 40, 80, 160}
  acquisition ∈ {ucb_2, ucb_01, ucb_10, ei, random}
  Fixed: batch=4, w=1.0

Both studies run on 4 benchmarks × 20 seeds, shared initialisations.
Benchmarks: hartmann6 (multimodal), pareto_20210112 (structured),
            pareto_20201218 (flat), coatings (smooth).

Expected runtime: Study 1 ~15 min, Study 2 (fractional) ~10 min,
                  Study 2 (full) ~2 hours. All non-LLM.

Usage:
    python ablation_batch_pool.py --study 1 --data_dir data/ --out_dir results_ablation2/
    python ablation_batch_pool.py --study 2 --data_dir data/ --out_dir results_ablation2/
    python ablation_batch_pool.py --study 2 --full --data_dir data/ --out_dir results_ablation2/
"""

import argparse
import logging
import pathlib
import sys
import warnings
import itertools

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from oracle import NNOracle
from evaluate import compute_metrics, aggregate_across_seeds
from shared_seed_experiment import generate_shared_inits, _save_and_print

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DATASETS = [
    ("hartmann6",       "hartmann6",                            "multimodal"),
    ("pareto_20210112", "pareto_campaign 2021-01-12_16-26-56",  "structured"),
    ("pareto_20201218", "pareto_campaign 2020-12-18_17-38-40",  "flat"),
    ("coatings",        "coatings",                             "smooth"),
]


# ── Acquisition functions ─────────────────────────────────────────────────────

def _score_ucb(mu, sigma, beta=2.0):
    return mu + beta * sigma

def _score_ei(mu, sigma, y_best):
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best) / sigma
    return sigma * (norm.pdf(z) + z * norm.cdf(z))

def _score_pi(mu, sigma, y_best, xi=0.01):
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-9)
    return norm.cdf((mu - y_best - xi) / sigma)

def _score_random(mu, sigma, **kwargs):
    return np.random.randn(len(mu))

ACQ_FNS = {
    "ucb_2":   lambda mu, sig, yb: _score_ucb(mu, sig, 2.0),
    "ucb_01":  lambda mu, sig, yb: _score_ucb(mu, sig, 0.1),
    "ucb_10":  lambda mu, sig, yb: _score_ucb(mu, sig, 10.0),
    "ei":      lambda mu, sig, yb: _score_ei(mu, sig, yb),
    "random":  lambda mu, sig, yb: _score_random(mu, sig),
}


# ── Core EGBO runner with configurable batch/pool/acq ────────────────────────

def run_egbo_configurable(
    oracle, X_init, y_init, budget,
    batch_size=4,
    re_fit_gp=False,
    ea_candidates=72,
    bo_candidates=8,
    acquisition="ucb_2",
    novelty_weight=1.0,   # 1.0 = pure acquisition (no novelty)
    random_state=0,
):
    """
    EGBO with fully configurable batch size, pool composition, and acquisition.

    Parameters
    ----------
    batch_size    : points selected per round
    re_fit_gp     : if True, refit GP between each selection within a batch
    ea_candidates : number of U-NSGA-III evolutionary candidates
    bo_candidates : number of random BO candidates (scored by acquisition)
    acquisition   : one of {ucb_2, ucb_01, ucb_10, ei, random}
    novelty_weight: w in score = w*acq + (1-w)*novelty. 1.0 = pure acq.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    warnings.filterwarnings("ignore")

    bounds = oracle.bounds()
    d = bounds.shape[0]
    N_dataset = len(oracle._X_raw)
    all_X = oracle._X_raw
    scaler_oracle = oracle._scaler

    lo, hi = bounds[:, 0], bounds[:, 1]
    def norm(X): return (X - lo) / (hi - lo + 1e-12)
    def denorm(Xn): return Xn * (hi - lo) + lo

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    # Snap init points to dataset indices
    X_all_s = scaler_oracle.transform(all_X)
    queried = set()
    for row in scaler_oracle.transform(X_init):
        queried.add(int(np.argmin(np.linalg.norm(X_all_s - row, axis=1))))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []
    pool_size = ea_candidates + bo_candidates
    acq_fn = ACQ_FNS.get(acquisition, ACQ_FNS["ucb_2"])

    n_batches = max(1, (budget - len(X_init)) // batch_size)

    def _fit_gp(X, y):
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
            alpha=1e-6, normalize_y=True, n_restarts_optimizer=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(Xs, y)
        return gp, sc

    def _get_candidates(X_obs_norm, y_obs_local, rng_state):
        rng = np.random.default_rng(rng_state)

        # EA candidates — skip NSGA-III if ea_candidates < 2 (pymoo requirement)
        if ea_candidates >= 2:
            top_k = min(ea_candidates, len(y_obs_local))
            top_idx = np.argsort(y_obs_local)[-top_k:][::-1]
            seed_x = X_obs_norm[top_idx]
            if seed_x.shape[0] < ea_candidates:
                pad = rng.random((ea_candidates - seed_x.shape[0], d))
                seed_x = np.vstack([seed_x, pad])
            try:
                # pop_size must be >= n_ref_dirs (always 1 for single-obj)
                pop_size = max(ea_candidates, 2)
                ref_dirs = get_reference_directions("energy", 1, pop_size, seed=rng_state)
                algo = UNSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=seed_x)
                pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                                  xl=np.zeros(d), xu=np.ones(d))
                algo.setup(pm, termination=NoTermination())
                pop = algo.ask()
                n_act = len(pop)
                f_vals = -y_obs_local[np.argsort(y_obs_local)[-n_act:]]
                pop.set("F", f_vals.reshape(-1, 1))
                algo.tell(infills=pop)
                ea_cands = np.clip(algo.ask().get("X"), 0, 1)[:ea_candidates]
            except Exception:
                ea_cands = rng.random((ea_candidates, d))
        else:
            ea_cands = np.zeros((0, d))  # no EA candidates

        # BO candidates (random grid — acquisition scores them)
        bo_cands = rng.random((bo_candidates, d)) if bo_candidates > 0 else np.zeros((0, d))

        parts = [p for p in [ea_cands, bo_cands] if len(p) > 0]
        return np.vstack(parts) if parts else rng.random((8, d))

    for batch_idx in range(n_batches):
        X_obs_norm = norm(X_obs)
        gp, sc = _fit_gp(X_obs, y_obs)

        candidates_norm = _get_candidates(X_obs_norm, y_obs,
                                          random_state + batch_idx)
        candidates_raw = denorm(candidates_norm)

        selected_indices = []
        remaining = list(range(len(candidates_norm)))

        for pick_idx in range(batch_size):
            if not remaining:
                break

            # Re-fit GP if requested (sequential mode)
            if re_fit_gp and pick_idx > 0 and selected_indices:
                extra_x = candidates_raw[selected_indices]
                extra_y = np.array([oracle.query(x) for x in extra_x])
                X_tmp = np.vstack([X_obs, extra_x])
                y_tmp = np.append(y_obs, extra_y)
                gp, sc = _fit_gp(X_tmp, y_tmp)

            # Score all remaining candidates
            rem = np.array(remaining)
            cands_rem_raw = candidates_raw[rem]
            cands_rem_s = sc.transform(cands_rem_raw)
            mu, sigma = gp.predict(cands_rem_s, return_std=True)
            y_best = y_obs.max()
            acq_scores = acq_fn(mu, sigma, y_best)

            # Novelty scores
            already = (candidates_norm[selected_indices]
                       if selected_indices else np.zeros((0, d)))
            obs_combined = (np.vstack([X_obs_norm, already])
                            if len(already) else X_obs_norm)
            nov = np.array([
                np.min(np.linalg.norm(obs_combined - candidates_norm[i], axis=1))
                for i in rem
            ])

            # Normalise and combine
            a = acq_scores
            a_norm = (a - a.min()) / (a.max() - a.min() + 1e-12)
            n_norm = (nov - nov.min()) / (nov.max() - nov.min() + 1e-12)
            score = novelty_weight * a_norm + (1 - novelty_weight) * n_norm

            pick = int(rem[np.argmax(score)])
            selected_indices.append(pick)
            remaining.remove(pick)

        # Snap to nearest unqueried dataset rows
        unqueried = [i for i in range(N_dataset) if i not in queried]
        if not unqueried:
            unqueried = list(range(N_dataset))

        new_x_rows, new_y_vals = [], []
        for idx in selected_indices:
            x_raw = candidates_raw[idx]
            x_s = scaler_oracle.transform(x_raw.reshape(1, -1))[0]
            pool_s = scaler_oracle.transform(all_X[unqueried])
            chosen = unqueried[int(np.argmin(np.linalg.norm(pool_s - x_s, axis=1)))]
            queried.add(chosen)
            new_x_rows.append(all_X[chosen])
            new_y_vals.append(float(oracle._y_raw[chosen]))
            unqueried = [i for i in unqueried if i != chosen]

        X_obs = np.vstack([X_obs, np.array(new_x_rows)])
        y_obs = np.append(y_obs, new_y_vals)
        for _ in new_y_vals:
            running_best.append(float(y_obs.max()))

        decisions.append((len(X_init) + batch_idx * batch_size,
                          f"egbo_b{batch_size}_ea{ea_candidates}", {}))

    return {"running_best": running_best, "decisions": decisions,
            "failures": [], "X_obs": X_obs, "y_obs": y_obs}


# ── Study 1: Batch size ───────────────────────────────────────────────────────

def study1_conditions():
    """All batch size × re_fit_gp combinations."""
    conditions = []
    for bs in [1, 2, 4, 8]:
        for refit in [False, True]:
            if bs == 1 and refit:
                continue  # re_fit meaningless for batch=1
            label = f"batch{bs}_{'refit' if refit else 'joint'}"
            conditions.append({
                "label": label, "batch_size": bs, "re_fit_gp": refit,
                "ea_candidates": 72, "bo_candidates": 8,
                "acquisition": "ucb_2", "novelty_weight": 1.0,
            })
    return conditions


# ── Study 2: Pool composition ─────────────────────────────────────────────────

def study2_conditions(full=False):
    """
    Fractional factorial (default) or full factorial (--full).
    Fractional: 8 conditions covering main effects of ea_ratio × pool_size.
    Full: all 5×4×5 = 100 conditions.
    """
    if not full:
        # 2^3 fractional factorial: ea_ratio × pool_size × acquisition
        # Covers main effects with only 8 runs
        return [
            {"label": "ea20_p40_ucb2",   "ea_ratio": 0.2, "pool_size": 40,  "acquisition": "ucb_2"},
            {"label": "ea20_p40_rand",    "ea_ratio": 0.2, "pool_size": 40,  "acquisition": "random"},
            {"label": "ea20_p160_ucb2",   "ea_ratio": 0.2, "pool_size": 160, "acquisition": "ucb_2"},
            {"label": "ea20_p160_rand",   "ea_ratio": 0.2, "pool_size": 160, "acquisition": "random"},
            {"label": "ea90_p40_ucb2",    "ea_ratio": 0.9, "pool_size": 40,  "acquisition": "ucb_2"},
            {"label": "ea90_p40_rand",    "ea_ratio": 0.9, "pool_size": 40,  "acquisition": "random"},
            {"label": "ea90_p160_ucb2",   "ea_ratio": 0.9, "pool_size": 160, "acquisition": "ucb_2"},
            {"label": "ea90_p160_rand",   "ea_ratio": 0.9, "pool_size": 160, "acquisition": "random"},
            # Reference: current EGBO default
            {"label": "egbo_default",     "ea_ratio": 0.9, "pool_size": 80,  "acquisition": "ucb_2"},
        ]
    else:
        conditions = []
        for ea_ratio in [0.0, 0.2, 0.5, 0.9, 1.0]:
            for pool_size in [20, 40, 80, 160]:
                for acq in ["ucb_2", "ucb_01", "ucb_10", "ei", "random"]:
                    ea_n = int(ea_ratio * pool_size)
                    bo_n = pool_size - ea_n
                    label = f"ea{int(ea_ratio*100)}_p{pool_size}_{acq}"
                    conditions.append({
                        "label": label, "ea_ratio": ea_ratio,
                        "pool_size": pool_size, "acquisition": acq,
                    })
        return conditions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",      type=int, default=1, choices=[1, 2])
    parser.add_argument("--full",       action="store_true",
                        help="Study 2: run full factorial (100 conds) vs fractional (8)")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--out_dir",    default="results_ablation2")
    parser.add_argument("--n_repeats",  type=int, default=20)
    parser.add_argument("--n_init",     type=int, default=5)
    parser.add_argument("--budget_frac",type=float, default=0.5)
    parser.add_argument("--datasets",   nargs="+", default=None)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_list = DATASETS
    if args.datasets:
        ds_list = [(l, n, r) for l, n, r in DATASETS if l in args.datasets]

    if args.study == 1:
        conditions = study1_conditions()
        study_label = "batch_size"
    else:
        conditions = study2_conditions(full=args.full)
        study_label = "pool_composition_full" if args.full else "pool_composition_frac"

    print(f"Study {args.study} ({study_label}): {len(conditions)} conditions")
    print(f"Datasets: {[d[0] for d in ds_list]}")
    print(f"Seeds: {args.n_repeats}")
    print(f"Total runs: {len(conditions) * len(ds_list) * args.n_repeats}")
    print()

    all_rows = []

    for ds_label, ds_name, ds_regime in ds_list:
        print(f"\n{'='*60}\nDataset: {ds_label} [{ds_regime}]")
        oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        N      = len(oracle._X_raw)
        budget = max(args.n_init + 10, int(args.budget_frac * N))
        gb     = oracle.global_best()
        gmin   = float(oracle._y_raw.min())
        print(f"  N={N}  budget={budget}  dims={oracle.bounds().shape[0]}")

        shared_inits = generate_shared_inits(
            oracle, args.n_repeats, args.n_init, rng_seed=42)
        best_norms = [y.max()/gb for _, y in shared_inits]
        print(f"  Init best_norm: {min(best_norms):.2f}–{max(best_norms):.2f}")

        for cond in conditions:
            label = cond["label"]
            seed_metrics, seed_curves, seed_logs = [], [], []

            # Build runner kwargs
            if args.study == 1:
                kwargs = {k: cond[k] for k in
                          ["batch_size","re_fit_gp","ea_candidates",
                           "bo_candidates","acquisition","novelty_weight"]}
            else:
                pool_size = cond["pool_size"]
                ea_ratio  = cond["ea_ratio"]
                # ea_n=0 is valid (pure BO); ea_n=1 would cause pymoo pop_size warning
                # so snap to 0 or >=2
                ea_n_raw = int(ea_ratio * pool_size)
                ea_n = 0 if ea_n_raw < 2 else ea_n_raw
                bo_n = max(0, pool_size - ea_n)
                kwargs = {
                    "batch_size": 4, "re_fit_gp": False,
                    "ea_candidates": ea_n, "bo_candidates": bo_n,
                    "acquisition": cond["acquisition"], "novelty_weight": 1.0,
                }

            for rep_idx, (X_init, y_init) in enumerate(shared_inits):
                try:
                    res = run_egbo_configurable(
                        oracle, X_init, y_init, budget,
                        random_state=rep_idx, **kwargs,
                    )
                    m = compute_metrics(res, gb, budget=budget, oracle_global_min=gmin)
                    seed_metrics.append(m)
                    seed_curves.append(np.array(res["running_best"]))
                    seed_logs.append({"decisions": res["decisions"], "failures": []})
                except Exception as e:
                    logger.warning(f"{label} rep {rep_idx}: {e}")

            _save_and_print(out_dir / study_label, ds_label, label,
                            seed_metrics, seed_curves, seed_logs,
                            gb, gmin, shared_inits, all_rows)
            # Add regime to rows
            for row in all_rows:
                if "regime" not in row:
                    row["regime"] = ds_regime

    df = pd.DataFrame(all_rows)
    # Add config columns
    if args.study == 1:
        df["batch_size"] = df["condition"].str.extract(r"batch(\d+)").astype(float)
        df["refit"]      = df["condition"].str.contains("refit")
    else:
        df["ea_ratio"]   = df["condition"].str.extract(r"ea(\d+)").astype(float) / 100
        df["pool_size"]  = df["condition"].str.extract(r"p(\d+)").astype(float)
        df["acq"]        = df["condition"].str.extract(r"_(ucb\w+|ei|rand\w+)$")

    csv_path = out_dir / f"{study_label}_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Quick results table ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {study_label}")
    print(f"{'='*60}")

    for ds_label, _, ds_regime in ds_list:
        sub = df[df["dataset"] == ds_label]
        if sub.empty:
            continue
        print(f"\n{ds_label} [{ds_regime}]:")
        agg = sub.groupby("condition")["auc_best"].agg(["mean","std","count"])
        agg = agg.sort_values("mean", ascending=False)
        for cond, row in agg.iterrows():
            print(f"  {cond:<35} {row['mean']:.3f}±{row['std']:.3f}  n={int(row['count'])}")

    # ── Key statistical test: does batch/pool effect depend on regime? ───────
    if args.study == 1 and "batch_size" in df.columns:
        print(f"\n{'='*60}")
        print("BATCH SIZE EFFECT BY REGIME")
        print(f"{'='*60}")
        for regime in df["regime"].dropna().unique():
            sub = df[df["regime"] == regime]
            bs_groups = [sub[sub["batch_size"] == bs]["auc_best"].values
                         for bs in [1, 2, 4, 8] if bs in sub["batch_size"].values]
            if len(bs_groups) >= 2:
                f_stat, p = stats.f_oneway(*bs_groups)
                print(f"  {regime:<15} F={f_stat:.2f} p={p:.3f}"
                      + (" * batch size matters" if p < 0.05 else ""))
                means = {bs: sub[sub["batch_size"] == bs]["auc_best"].mean()
                         for bs in [1, 2, 4, 8]
                         if bs in sub["batch_size"].values}
                print(f"    " + "  ".join(f"b{bs}={m:.3f}" for bs,m in sorted(means.items())))


if __name__ == "__main__":
    main()
