"""
generate_bo_diagnostics.py — lightweight (sklearn-GP-only, no torch/botorch)
replication of the gen6_child0_tuned (front-range-normalised UCB) vs
hint_fixed_ucb (raw-sigma UCB) comparison on TunableSyntheticMOOracle, built
to produce the JSON a visualisation of "what's happening in BO" can be built
from — specifically the front-range-shrinkage mechanism track_front_range.py
was written to test, plus the HV-convergence outcome, plus the domain-
selection sweep from sweep_tunable_domain.py.

This is deliberately NOT the full evolution harness (full_replay.py /
excipient_campaign_mo.py use botorch, unavailable in this environment) — it's
a self-contained stand-in using the same two AF formulas, same oracle, same
batch-UCB idea (greedy top-k by score, no joint batch acquisition), run with
sklearn GPs. Good enough to check the MECHANISM (does front_range shrink?
does the normalised AF's effective beta move?) even if the exact HV numbers
won't match a qNEHVI-driven campaign.

Usage:
    python generate_bo_diagnostics.py --out bo_diagnostics.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
# tunable_synthetic_oracle.py moved to shared/ this session (so
# full_replay.py, which lives there, can import it) — added here so this
# pre-existing script keeps working rather than shadowing/duplicating it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "shared"))
from tunable_synthetic_oracle import TunableSyntheticMOOracle  # noqa: E402

HERE = pathlib.Path(__file__).parent

RAW_BETA = 2.0
NORM_BETA = 15.0


def fit_gp(X, y):
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1)
    gp.fit(X, y)
    return gp


def pareto_front_mask(Y, directions):
    # both-minimise convention throughout this codebase
    n = len(Y)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        le = np.all(Y <= Y[i], axis=1)
        lt = np.any(Y < Y[i], axis=1)
        beats = le & lt
        beats[i] = False
        if beats.any():
            dominated[i] = True
    return ~dominated


def front_range(Y, directions):
    mask = pareto_front_mask(Y, directions)
    front = Y[mask]
    if len(front) < 2:
        return np.ptp(Y, axis=0)
    return np.ptp(front, axis=0)


def hypervolume_2d(Y, ref):
    """Exact 2-objective hypervolume (both-minimise) w.r.t. reference point."""
    mask = pareto_front_mask(Y, None)
    front = Y[mask]
    front = front[front[:, 0].argsort()]
    hv = 0.0
    prev_f1 = ref[0]
    # sweep from best (lowest) f1 to worst; standard 2D HV via sorted strip sum
    for f1, f2 in front[::-1]:
        if f1 >= ref[0] or f2 >= ref[1]:
            continue
        hv += (prev_f1 - f1) * (ref[1] - f2)
        prev_f1 = f1
    return float(hv)


def run_condition(oracle, X_init, Y_init, budget, batch_size, condition, rng):
    directions = oracle.objective_directions()
    names = oracle.objective_names()
    X_obs, Y_obs = X_init.copy(), Y_init.copy()
    n_batches = max(1, (budget - len(X_init)) // batch_size)
    remaining = set(range(len(oracle))) - set(oracle._queried)

    ref = oracle._Y_true.max(axis=0) * 1.1

    trace = []
    fr0 = front_range(Y_obs, directions)
    trace.append({
        "batch": 0, "n_obs": len(Y_obs),
        "front_range": {n: float(v) for n, v in zip(names, fr0)},
        "hv": hypervolume_2d(Y_obs, ref),
    })

    for b in range(n_batches):
        gp1 = fit_gp(X_obs, Y_obs[:, 0])
        gp2 = fit_gp(X_obs, Y_obs[:, 1])
        pool_idx = np.array(sorted(remaining))
        X_pool = oracle._X_raw[pool_idx]
        mu1, std1 = gp1.predict(X_pool, return_std=True)
        mu2, std2 = gp2.predict(X_pool, return_std=True)
        mu_sum = mu1 + mu2

        fr = front_range(Y_obs, directions)
        fr_safe = np.clip(fr, 1e-6, None)

        if condition == "raw_ucb":
            sigma_term = std1 + std2
            beta = RAW_BETA
        else:  # front_range_norm
            sigma_term = std1 / fr_safe[0] + std2 / fr_safe[1]
            beta = NORM_BETA
        # both AFs MINIMISE mu_sum but want HIGH uncertainty explored ->
        # score to minimise: mu_sum - beta*sigma_term (matches the repo's
        # score_pool convention where higher score = better candidate for a
        # maximiser; here we sort ascending on mu_sum - beta*sigma so it's
        # equivalent for a min-min problem)
        score = mu_sum - beta * sigma_term
        order = np.argsort(score)[:batch_size]
        chosen = pool_idx[order]

        new_x = oracle._X_raw[chosen]
        new_y = np.array([oracle.query_mo(oracle._X_raw[i])[0] for i in chosen])
        remaining -= set(chosen.tolist())

        X_obs = np.vstack([X_obs, new_x])
        Y_obs = np.vstack([Y_obs, new_y])

        fr_after = front_range(Y_obs, directions)
        eff_beta = {n: float(beta / fr_after[i]) if condition == "front_range_norm" else float(beta)
                    for i, n in enumerate(names)}
        trace.append({
            "batch": b + 1, "n_obs": len(Y_obs),
            "front_range": {n: float(v) for n, v in zip(names, fr_after)},
            "effective_beta": eff_beta,
            "hv": hypervolume_2d(Y_obs, ref),
        })

    pf_mask = pareto_front_mask(Y_obs, directions)
    return trace, Y_obs, pf_mask


def sweep_grid():
    """Reproduces sweep_tunable_domain.py's dominance_ratio/achieved_cv/
    front_range_ratio grid (duplicated here in sklearn-only form so this
    script has no cross-file coupling)."""
    N_TRAIN, N_REPS = 30, 3
    plateau_grid = [1.0, 2.0, 3.0, 5.0]
    noise_grid = [0.03, 0.08, 0.15]
    rows = []
    for plateau in plateau_grid:
        for noise_level in noise_grid:
            doms, cvs, rrs = [], [], []
            for r in range(N_REPS):
                seed = 42 + r
                oracle = TunableSyntheticMOOracle.build(
                    plateau_sharpness=plateau, noise_level=noise_level,
                    noise_mode="proportional", seed=seed)
                rng = np.random.default_rng(seed + 100)
                train_idx = rng.choice(len(oracle), size=N_TRAIN, replace=False)
                X_train, Y_noisy, cv_samples = [], [], []
                for idx in train_idx:
                    x = oracle._X_raw[idx]
                    y_noisy, chosen = oracle.query_mo(x)
                    f_true = oracle.true_y(chosen)
                    X_train.append(x)
                    Y_noisy.append(y_noisy)
                    cv_samples.append(np.abs(y_noisy - f_true) / (np.abs(f_true) + 1e-9))
                X_train, Y_noisy = np.array(X_train), np.array(Y_noisy)
                gp1, gp2 = fit_gp(X_train, Y_noisy[:, 0]), fit_gp(X_train, Y_noisy[:, 1])
                remaining = [i for i in range(len(oracle)) if i not in oracle._queried]
                X_pool = oracle._X_raw[remaining]
                mu1, std1 = gp1.predict(X_pool, return_std=True)
                mu2, std2 = gp2.predict(X_pool, return_std=True)
                fr = np.clip(front_range(Y_noisy, None), 1e-6, None)
                mu_sum = mu1 + mu2
                sigma_norm = std1 / fr[0] + std2 / fr[1]
                doms.append(np.std(mu_sum) / (np.std(sigma_norm) + 1e-9))
                cvs.append(float(np.mean(cv_samples)))
                rrs.append(float(fr[1] / fr[0]))
            rows.append({
                "plateau_sharpness": plateau, "noise_level": noise_level,
                "dominance_ratio": float(np.mean(doms)),
                "achieved_cv": float(np.mean(cvs)),
                "front_range_ratio": float(np.mean(rrs)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_campaigns", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--plateau_sharpness", type=float, default=5.0)
    ap.add_argument("--noise_level", type=float, default=0.08)
    ap.add_argument("--noise_mode", default="proportional")
    ap.add_argument("--scale2", type=float, default=3.0)
    ap.add_argument("--out", default=str(HERE / "bo_diagnostics.json"))
    args = ap.parse_args()

    names_ref = ["f1", "f2"]
    conditions = ["raw_ucb", "front_range_norm"]
    all_traces = {c: [] for c in conditions}
    snapshots = {c: [] for c in conditions}  # per-campaign final Y_obs + pf mask, campaign 0 only

    for i in range(args.n_campaigns):
        oracle = TunableSyntheticMOOracle.build(
            plateau_sharpness=args.plateau_sharpness, noise_level=args.noise_level,
            noise_mode=args.noise_mode, scale2=args.scale2, seed=42 + i)
        rng = np.random.default_rng(1000 + i)
        init_idx = rng.choice(len(oracle), size=args.n_init, replace=False)
        X_init = oracle._X_raw[init_idx]
        Y_init = np.array([oracle.query_mo(x)[0] for x in X_init])
        names = oracle.objective_names()

        for cond in conditions:
            # fresh oracle copy per condition (same seed -> same pool/noise draw)
            oracle_c = TunableSyntheticMOOracle.build(
                plateau_sharpness=args.plateau_sharpness, noise_level=args.noise_level,
                noise_mode=args.noise_mode, scale2=args.scale2, seed=42 + i)
            oracle_c._queried = set(init_idx.tolist())
            trace, Y_obs, pf_mask = run_condition(
                oracle_c, X_init, Y_init, args.budget, args.batch_size, cond, rng)
            all_traces[cond].append(trace)
            if i == 0:
                snapshots[cond] = {
                    "Y_obs": Y_obs.tolist(),
                    "pf_mask": pf_mask.tolist(),
                }

    print("Running domain-selection sweep grid (sklearn-only reproduction)...")
    grid = sweep_grid()

    out = {
        "meta": {
            "n_campaigns": args.n_campaigns, "budget": args.budget,
            "n_init": args.n_init, "batch_size": args.batch_size,
            "plateau_sharpness": args.plateau_sharpness, "noise_level": args.noise_level,
            "noise_mode": args.noise_mode, "scale2": args.scale2,
            "objective_names": names_ref,
            "raw_beta": RAW_BETA, "norm_beta": NORM_BETA,
        },
        "traces": all_traces,
        "campaign0_snapshot": snapshots,
        "sweep_grid": grid,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")

    # quick console summary
    for cond, traces in all_traces.items():
        fr0 = {n: np.mean([t[0]["front_range"][n] for t in traces]) for n in names_ref}
        frN = {n: np.mean([t[-1]["front_range"][n] for t in traces]) for n in names_ref}
        hv0 = np.mean([t[0]["hv"] for t in traces])
        hvN = np.mean([t[-1]["hv"] for t in traces])
        print(f"{cond}: front_range {fr0}->{frN}  HV {hv0:.3f}->{hvN:.3f}")


if __name__ == "__main__":
    main()
