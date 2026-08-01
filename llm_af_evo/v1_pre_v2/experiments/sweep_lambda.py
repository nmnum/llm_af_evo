"""
sweep_lambda.py — llm_af_evo pre-test, step 4: tune the exploration-credit
term's lambda so a purely GP-based fitness (no oracle) separates EGBO-
novelty's actual pick from a pure-exploitation counterfactual at the same
rate as the true-oracle-HV diagnostic did.

fitness(batch) = HV_pred_improvement(batch) + lambda * sigma_norm(batch)

  HV_pred_improvement(batch) = HV(front U {mu(x) : x in batch}) - HV(front)
    [GP posterior mean mu, deterministic, all-maximise convention]
  sigma_norm(batch) = sum over x in batch of sum_j sigma_j(x) / range_j
    [range_j = observed-data range of objective j so far, per-campaign,
     online-adaptive; sum-over-objectives aggregation, per the design
     decision — max-j is the documented fallback if this underperforms]

Both terms use the GP posterior only (mu, sigma) — no oracle ground truth —
so this fitness is deployable on real data, unlike the true-oracle-HV
diagnostic it's being validated against.

Selection criterion: lambda* = argmax win_rate(lambda), where win_rate is
the fraction of steps where fitness(egbo_pick) > fitness(exploit_pick).
Validation: win_rate(lambda*) should be >= the diagnostic's 66.7% true-
oracle-HV win rate. Spearman correlation between fitness and true-oracle-HV
gain is reported as a secondary sanity check, not the selection target.

Also reports the sigma_norm-vs-novelty correlation across all logged pool
candidates, per the documented risk: if GP uncertainty and input-space
novelty (what mo_egbo_novelty's own selection already uses) are highly
correlated, the evolution's search space for a genuinely new AF is thin.

Usage:
    python sweep_lambda.py --logs_dir logs
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

from diagnostic_true_oracle_hv import (
    to_allmax, hv_of, nearest_true_y, greedy_exploit_batch,
    OBJECTIVE_DIRECTIONS,
)

LAMBDA_GRID = [0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000,
               5000, 10000, 20000, 50000, 100000]


def sigma_norm_of_set(sigma_set: np.ndarray, range_j: np.ndarray) -> float:
    """sum over candidates in the set of (sum over objectives of sigma_j/range_j)."""
    return float(np.sum(sigma_set / range_j))


def analyse_seed_log(log: dict) -> list:
    """
    Returns per-step records with everything needed to compute fitness at
    any lambda without re-touching the oracle: mu/sigma for both the actual
    egbo batch and the exploitation-only counterfactual batch, the front
    HV, range_j, and (for the correlation check) each step's already-known
    true-oracle HV gains from the diagnostic's own logic.
    """
    oracle_X = np.array(log["oracle_X_raw"])
    oracle_Y = np.array(log["oracle_Y_raw"])
    Y_all_allmax = to_allmax(oracle_Y)
    ref_point_allmax = (Y_all_allmax.min(axis=0) -
                         0.1 * (Y_all_allmax.max(axis=0) - Y_all_allmax.min(axis=0) + 1e-9))

    Y_running = np.array(log["Y_init"])
    X_running = np.array(log["X_init"])
    batch_size = log["batch_size"]
    rows = []

    for step in log["decisions"]:
        if "pool_x_norm" not in step or "pool_pred_sigma" not in step:
            continue

        front_allmax = to_allmax(Y_running.copy())
        front_hv = hv_of(front_allmax, ref_point_allmax)
        range_j = np.ptp(Y_running, axis=0)
        range_j = np.maximum(range_j, 1e-6)

        pool_x = np.array(step["pool_x_norm"])
        pool_mu = np.array(step["pool_pred_mu"])       # all-max convention
        pool_sigma = np.array(step["pool_pred_sigma"])  # same convention, nonneg

        egbo_idx = step["pool_selected_idx"]
        exploit_idx = greedy_exploit_batch(pool_mu, front_allmax, ref_point_allmax, batch_size)

        def hv_pred_gain(idx_list):
            mu_set = pool_mu[idx_list]
            return hv_of(np.vstack([front_allmax, mu_set]), ref_point_allmax) - front_hv

        egbo_hv_pred = hv_pred_gain(egbo_idx)
        exploit_hv_pred = hv_pred_gain(exploit_idx)
        egbo_sigma_norm = sigma_norm_of_set(pool_sigma[egbo_idx], range_j)
        exploit_sigma_norm = sigma_norm_of_set(pool_sigma[exploit_idx], range_j)

        # True-oracle HV gains, same logic as the diagnostic, needed only
        # for the Spearman sanity check (not for win-rate/lambda selection).
        actual_true_y = np.array(step["picked_y"])
        egbo_true_gain = hv_of(np.vstack([front_allmax, to_allmax(actual_true_y)]),
                                ref_point_allmax) - front_hv
        exploit_true_y = np.array([nearest_true_y(pool_x[i], oracle_X, oracle_Y)
                                    for i in exploit_idx])
        exploit_true_gain = hv_of(np.vstack([front_allmax, to_allmax(exploit_true_y)]),
                                   ref_point_allmax) - front_hv

        # sigma-vs-novelty correlation inputs: novelty(x) = nearest-neighbour
        # distance from x to everything observed so far, in the same
        # normalised [0,1]^16 space pool_x_norm already lives in.
        novelty_all = np.array([
            np.min(np.linalg.norm(X_running - pool_x[i], axis=1))
            for i in range(len(pool_x))
        ])
        sigma_norm_all = np.array([
            float(np.sum(pool_sigma[i] / range_j)) for i in range(len(pool_x))
        ])

        rows.append({
            "egbo_hv_pred": egbo_hv_pred, "exploit_hv_pred": exploit_hv_pred,
            "egbo_sigma_norm": egbo_sigma_norm, "exploit_sigma_norm": exploit_sigma_norm,
            "egbo_true_gain": egbo_true_gain, "exploit_true_gain": exploit_true_gain,
            "novelty_all": novelty_all, "sigma_norm_all": sigma_norm_all,
        })

        Y_running = np.vstack([Y_running, actual_true_y])
        X_running = np.vstack([X_running, np.array(step["picked_x"])])

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default=str(pathlib.Path(__file__).parent / "logs"))
    args = ap.parse_args()
    logs_dir = pathlib.Path(args.logs_dir)

    all_rows = []
    for cond_dir in sorted(logs_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        for f in sorted(cond_dir.glob("seed_*.json")):
            with open(f) as fh:
                log = json.load(fh)
            rows = analyse_seed_log(log)
            for r in rows:
                r["condition"] = log["condition"]
                r["seed"] = log["seed"]
            all_rows.extend(rows)

    if not all_rows:
        print("No diagnosable steps found.")
        return

    n = len(all_rows)
    print(f"Steps analysed: {n}\n")
    print(f"{'lambda':>8}  {'win_rate':>9}  {'spearman_r':>11}")

    # per-step win/loss flags per lambda, kept per (condition, seed) so we
    # can report win-rate variance across seeds, not just the pooled number
    # — a pooled win-rate can look like a stable margin while actually
    # being a coin flip that happened to land right, if per-seed win-rates
    # are all over the place.
    keys = sorted({(r["condition"], r["seed"]) for r in all_rows})

    best_lambda, best_win_rate = None, -1.0
    sweep_results = []
    for lam in LAMBDA_GRID:
        wins = 0
        fitness_vals, true_hv_vals = [], []
        per_seed_wins = {k: [] for k in keys}
        for r in all_rows:
            f_egbo = r["egbo_hv_pred"] + lam * r["egbo_sigma_norm"]
            f_exploit = r["exploit_hv_pred"] + lam * r["exploit_sigma_norm"]
            win = f_egbo > f_exploit
            wins += win
            per_seed_wins[(r["condition"], r["seed"])].append(win)
            fitness_vals += [f_egbo, f_exploit]
            true_hv_vals += [r["egbo_true_gain"], r["exploit_true_gain"]]
        win_rate = wins / n
        corr, _ = spearmanr(fitness_vals, true_hv_vals)
        per_seed_rates = np.array([np.mean(v) for v in per_seed_wins.values()])
        sweep_results.append((lam, win_rate, corr, per_seed_rates))
        print(f"{lam:>8}  {win_rate:>9.3f}  {corr:>11.3f}")
        if win_rate > best_win_rate:
            best_lambda, best_win_rate = lam, win_rate

    best_corr = next(c for l, w, c, _ in sweep_results if l == best_lambda)
    best_per_seed = next(p for l, w, c, p in sweep_results if l == best_lambda)
    print(f"\nPer-seed win-rate at lambda*={best_lambda}: "
          f"mean={best_per_seed.mean():.3f}  std={best_per_seed.std():.3f}  "
          f"min={best_per_seed.min():.3f}  max={best_per_seed.max():.3f}  "
          f"(n_seeds={len(best_per_seed)})")

    # keys and best_per_seed share index order (per_seed_wins was built as
    # {k: [] for k in keys}, and dicts preserve insertion order) — group by
    # condition to see whether the wide per-seed spread above tracks the
    # same warm-start split the true-oracle-HV diagnostic already showed
    # (mo_egbo_novelty 72.2% vs mo_ls_na_egbo 61.1%), which would be an
    # explainable pattern, versus scattered across both conditions, which
    # would mean the fitness is just noisy per-campaign rather than
    # systematically explainable.
    print(f"\nPer-seed win-rate at lambda*={best_lambda}, by condition:")
    by_cond = {}
    for (cond, seed), rate in zip(keys, best_per_seed):
        by_cond.setdefault(cond, []).append(rate)
    for cond, rates in sorted(by_cond.items()):
        rates = np.array(rates)
        print(f"  {cond:<18} mean={rates.mean():.3f}  std={rates.std():.3f}  "
              f"min={rates.min():.3f}  max={rates.max():.3f}  (n_seeds={len(rates)})")
    print(f"\nlambda* = {best_lambda}   win_rate(lambda*) = {best_win_rate:.3f}   "
          f"spearman(lambda*) = {best_corr:.3f}")

    diagnostic_win_rate = 120 / 180  # from the prior true-oracle-HV run; override if different
    if best_win_rate >= diagnostic_win_rate:
        print(f"\n=> PASS: win_rate(lambda*) >= diagnostic win rate "
              f"({diagnostic_win_rate:.3f}). Exploration-credit fitness is sound "
              f"— proceed to L2 evolution using this form and lambda*={best_lambda}.")
    else:
        print(f"\n=> SHORTFALL: win_rate(lambda*) < diagnostic win rate "
              f"({diagnostic_win_rate:.3f}). Try max-j aggregation for sigma_norm "
              f"before falling back further.")

    # sigma-vs-novelty correlation, pooled across every logged pool candidate
    all_novelty = np.concatenate([r["novelty_all"] for r in all_rows])
    all_sigma_norm = np.concatenate([r["sigma_norm_all"] for r in all_rows])
    sn_corr, _ = spearmanr(all_novelty, all_sigma_norm)
    print(f"\nsigma_norm vs novelty (input-space distance) Spearman correlation: "
          f"{sn_corr:.3f}")
    if sn_corr > 0.9:
        print("=> WARNING: GP uncertainty and novelty are near-collinear here — "
              "L2's evolution search space may be thin (evolved AF likely "
              "converges to a re-tuned version of the existing novelty-weighted "
              "baseline rather than something structurally new).")
    else:
        print("=> GP uncertainty and novelty are meaningfully distinct signals — "
              "evolution has room to discover something the baseline can't express.")

    # Persist everything make_figures.py (or any later analysis) needs —
    # previously print-only, so a past run's numbers were unrecoverable
    # once the terminal scrollback was gone.
    summary = {
        "n_steps": n,
        "diagnostic_win_rate_ref": diagnostic_win_rate,
        "sweep": [{"lambda": l, "win_rate": w, "spearman": c} for l, w, c, _ in sweep_results],
        "best_lambda": best_lambda, "best_win_rate": best_win_rate, "best_corr": best_corr,
        "per_seed_win_rate_at_best_lambda": {
            f"{cond}_seed{seed}": float(rate) for (cond, seed), rate in zip(keys, best_per_seed)
        },
        "per_seed_by_condition": {
            cond: {"mean": float(np.mean(rates)), "std": float(np.std(rates)),
                   "min": float(np.min(rates)), "max": float(np.max(rates))}
            for cond, rates in by_cond.items()
        },
        "sigma_norm_vs_novelty_spearman": float(sn_corr),
    }
    out_path = pathlib.Path(args.logs_dir).parent / "sweep_lambda_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
