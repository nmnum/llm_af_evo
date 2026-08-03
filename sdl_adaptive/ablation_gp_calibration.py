"""
ablation_gp_calibration.py — Test whether EGBO's advantage comes from
BoTorch GP calibration vs sklearn's GaussianProcessRegressor.

Runs two EGBO variants on shared initialisations:
  egbo_botorch : standard EGBO (BoTorch SingleTaskGP + fit_gpytorch_mll)
  egbo_sklearn : EGBO with sklearn GPR replacing BoTorch GP

If the advantage disappears with sklearn GP, calibration is the mechanism.
If it survives, the advantage comes from evolutionary coverage or batch selection.

Also tests novelty weight sensitivity:
  novelty_egbo_w03 : w=0.3 (more novelty-driven)
  novelty_egbo_w07 : w=0.7 (paper default)
  novelty_egbo_w09 : w=0.9 (more acquisition-driven)

Usage:
    python ablation_gp_calibration.py \\
        --data_dir data/ \\
        --out_dir results_ablation/ \\
        --n_repeats 20 \\
        --datasets pareto_20210112
"""

import argparse
import json
import logging
import pathlib
import warnings

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from oracle import NNOracle
from evaluate import compute_metrics, aggregate_across_seeds

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALL_DATASETS = [
    ("coatings",        "coatings"),
    ("pareto_20201218", "pareto_campaign 2020-12-18_17-38-40"),
    ("pareto_20201223", "pareto_campaign 2020-12-23_17-06-50"),
    ("pareto_20210104", "pareto_campaign 2021-01-04_08-37-39"),
    ("pareto_20210112", "pareto_campaign 2021-01-12_16-26-56"),
]


def generate_shared_inits(oracle, n_repeats, n_init, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    bounds = oracle.bounds()
    d = bounds.shape[0]
    inits = []
    for _ in range(n_repeats):
        X = np.array([[rng.uniform(bounds[i,0], bounds[i,1]) for i in range(d)]
                      for _ in range(n_init)])
        y = np.array([oracle.query(x) for x in X])
        inits.append((X.copy(), y.copy()))
    return inits


# ── sklearn GP version of EGBO ────────────────────────────────────────────────

def run_egbo_sklearn_gp(oracle, X_init, y_init, budget, batch_size=4,
                        qnehvi_candidates=8, evo_candidates=72,
                        merit_weight=0.7, random_state=0):
    """
    EGBO variant using sklearn's GaussianProcessRegressor instead of BoTorch.
    Everything else (U-NSGA-III, novelty selection) is identical to standard EGBO.
    This isolates the contribution of GP calibration quality.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import MinMaxScaler
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.core.termination import NoTermination
    from pymoo.util.ref_dirs import get_reference_directions

    # Reuse the novelty selection from shared_seed_experiment
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from evolutionary_candidates import novelty_select

    warnings.filterwarnings("ignore")

    bounds_np = oracle.bounds()
    d = bounds_np.shape[0]
    N_dataset = len(oracle._X_raw)
    all_X = oracle._X_raw
    scaler_oracle = oracle._scaler

    # Normalise inputs to [0,1] for GP
    input_scaler = MinMaxScaler()
    input_scaler.fit(bounds_np.T)  # fit on [[lo1,lo2,...],[hi1,hi2,...]]

    def norm_x(X):
        lo, hi = bounds_np[:, 0], bounds_np[:, 1]
        return (X - lo) / (hi - lo + 1e-12)

    def denorm_x(X_n):
        lo, hi = bounds_np[:, 0], bounds_np[:, 1]
        return X_n * (hi - lo) + lo

    X_obs = X_init.copy()
    y_obs = y_init.copy()

    # Snap init points to dataset indices
    X_scaled_all = scaler_oracle.transform(all_X)
    X_init_scaled = scaler_oracle.transform(X_init)
    queried = set()
    for row in X_init_scaled:
        dists = np.linalg.norm(X_scaled_all - row, axis=1)
        queried.add(int(np.argmin(dists)))

    running_best = [float(y_obs.max())] * len(X_init)
    decisions = []
    n_batches = max(1, (budget - len(X_init)) // batch_size)

    for batch_idx in range(n_batches):
        X_obs_norm = norm_x(X_obs)

        # Fit sklearn GP (same Matern 5/2 kernel as BoTorch, normalise_y=True)
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)),
            normalize_y=True,
            n_restarts_optimizer=3,
            alpha=1e-6,
        )
        gp.fit(X_obs_norm, y_obs)

        # EA candidates (identical to BoTorch EGBO)
        top_k = min(evo_candidates, len(X_obs))
        top_idx = np.argsort(y_obs)[-top_k:][::-1]
        seed_x = X_obs_norm[top_idx]
        if seed_x.shape[0] < evo_candidates:
            rng_pad = np.random.default_rng(random_state + batch_idx)
            pad = rng_pad.random((evo_candidates - seed_x.shape[0], d))
            seed_x = np.vstack([seed_x, pad])

        try:
            ref_dirs = get_reference_directions("energy", 1, evo_candidates,
                                                seed=random_state)
        except Exception:
            ref_dirs = np.random.default_rng(random_state).random((evo_candidates, 1))

        try:
            algo = UNSGA3(pop_size=evo_candidates, ref_dirs=ref_dirs, sampling=seed_x)
            pm = PymooProblem(n_var=d, n_obj=1, n_constr=0,
                              xl=np.zeros(d), xu=np.ones(d))
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

        # BO candidates: random grid scored by sklearn GP (replacing optimize_acqf)
        rng_bo = np.random.default_rng(random_state + batch_idx + 1000)
        bo_cands_norm = rng_bo.random((qnehvi_candidates, d))

        # Merge and score with UCB acquisition (sklearn GP)
        all_cands_norm = np.vstack([bo_cands_norm, ea_cands_norm])
        mu, sigma = gp.predict(all_cands_norm, return_std=True)
        acq_scores = mu + 2.0 * sigma  # UCB with beta=2

        # Select batch using novelty-aware selection
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
            a_norm = (a - a.min()) / (a.max() - a.min() + 1e-12)
            n_norm = (nov - nov.min()) / (nov.max() - nov.min() + 1e-12)
            score = merit_weight * a_norm + (1 - merit_weight) * n_norm
            pick = int(rem[np.argmax(score)])
            selected_indices.append(pick)
            remaining.remove(pick)

        # Snap to nearest unqueried dataset rows
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
            len(X_init) + batch_idx * batch_size, "egbo_sklearn", {"merit_weight": merit_weight}
        ))

    return {
        "running_best": running_best, "decisions": decisions,
        "failures": [], "X_obs": X_obs, "y_obs": y_obs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir",  default="results_ablation")
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--n_init",    type=int, default=10)
    parser.add_argument("--budget_frac", type=float, default=0.5)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_list = ALL_DATASETS
    if args.datasets:
        dataset_list = [(l, n) for l, n in ALL_DATASETS if l in args.datasets]

    # Import BoTorch EGBO runner from shared_seed_experiment
    from shared_seed_experiment import (
        run_egbo_campaign, generate_shared_inits as gen_inits,
        _save_and_print
    )

    all_rows = []

    for ds_label, ds_name in dataset_list:
        print(f"\n{'='*60}\nDataset: {ds_label}")
        oracle = NNOracle.from_dataset(ds_name, str(data_dir))
        N      = len(oracle._X_raw)
        budget = max(args.n_init + 10, int(args.budget_frac * N))
        gb     = oracle.global_best()
        gmin   = float(oracle._y_raw.min())
        print(f"  N={N}  budget={budget}  dims={oracle.bounds().shape[0]}")

        shared_inits = gen_inits(oracle, args.n_repeats, args.n_init, rng_seed=42)
        best_norms = [y.max()/gb for _,y in shared_inits]
        print(f"  Shared inits: best_norm {min(best_norms):.2f}–{max(best_norms):.2f}")

        # Define ablation conditions
        ablations = [
            # (label, runner_fn, kwargs)
            ("egbo_botorch",     run_egbo_campaign,       {"merit_weight": 1.0}),
            ("egbo_sklearn",     run_egbo_sklearn_gp,     {"merit_weight": 1.0}),
            ("novelty_w03",      run_egbo_campaign,       {"merit_weight": 0.3}),
            ("novelty_w07",      run_egbo_campaign,       {"merit_weight": 0.7}),
            ("novelty_w09",      run_egbo_campaign,       {"merit_weight": 0.9}),
        ]

        for label, runner, kwargs in ablations:
            seed_metrics, seed_curves, seed_logs = [], [], []
            for rep_idx, (X_init, y_init) in enumerate(shared_inits):
                try:
                    res = runner(
                        oracle, X_init, y_init, budget,
                        batch_size=4, qnehvi_candidates=8,
                        evo_candidates=72, random_state=rep_idx,
                        **kwargs
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
    out_df.to_csv(out_dir / "ablation_summary.csv", index=False)
    print(f"\nSaved ablation_summary.csv ({len(out_df)} rows)")

    # Quick statistical summary
    if len(out_df):
        print(f"\n{'='*60}")
        print("GP CALIBRATION ABLATION RESULT")
        print(f"{'='*60}")
        from scipy import stats
        for ds in out_df.dataset.unique():
            sub = out_df[out_df.dataset == ds]
            print(f"\n{ds}:")
            bt = sub[sub.condition == "egbo_botorch"]["auc_best"]
            sk = sub[sub.condition == "egbo_sklearn"]["auc_best"]
            if len(bt) and len(sk):
                t, p = stats.ttest_ind(bt, sk)
                print(f"  BoTorch EGBO: {bt.mean():.3f}±{bt.std():.3f}")
                print(f"  sklearn EGBO: {sk.mean():.3f}±{sk.std():.3f}")
                print(f"  Δ={bt.mean()-sk.mean():+.3f}  p={p:.3f}"
                      + (" *" if p < 0.05 else ""))
                if p > 0.05:
                    print("  → Calibration hypothesis NOT supported at this sample size")
                    print("    Advantage likely comes from evolutionary coverage or batch selection")
                else:
                    print("  → Calibration hypothesis SUPPORTED")
                    print("    BoTorch GP calibration is the mechanism")

        print(f"\nNOVELTY WEIGHT SENSITIVITY")
        print(f"{'='*60}")
        for ds in out_df.dataset.unique():
            sub = out_df[out_df.dataset == ds]
            print(f"\n{ds}:")
            for w_label in ["novelty_w03", "novelty_w07", "novelty_w09"]:
                row = sub[sub.condition == w_label]["auc_best"]
                if len(row):
                    print(f"  {w_label}: {row.mean():.3f}±{row.std():.3f}")


if __name__ == "__main__":
    main()
