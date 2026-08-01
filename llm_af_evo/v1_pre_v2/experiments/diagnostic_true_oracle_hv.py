"""
diagnostic_true_oracle_hv.py — llm_af_evo pre-test, step 3: true-oracle-HV
diagnostic that disambiguates failure mode A (predicted-vs-actual gap, fixed
by an exploration-credit term) from failure mode B (single-step regret is
fundamentally wrong, requires falling back to full counterfactual replay).

For every logged batch (from run_mock_subset.py's output), this script:
  1. Reconstructs the Pareto front as it stood BEFORE that batch's picks.
  2. Builds a "pure exploitation" counterfactual batch: the batch_size pool
     candidates that greedily maximise GP-PREDICTED hypervolume improvement
     (no exploration term at all — always trust the GP posterior mean).
  3. Looks up the TRUE oracle y for both the actual EGBO-novelty batch and
     the exploitation-only counterfactual batch (nearest-neighbour against
     the seed's dumped oracle pool — no refitting, no oracle-budget
     consumption, this is retrospective).
  4. Compares true-oracle HV improvement: actual (EGBO-novelty) vs
     exploitation-only.

Decision rule (pre-committed, see plan discussion):
  EGBO-novelty wins on true-oracle HV in most steps -> mode A -> proceed to
    build + tune the exploration-credit term.
  Pure exploitation ties or wins even with ground truth -> mode B -> single-
    step counterfactual regret is not a valid fitness signal; fall back to
    full counterfactual replay (2b) for any AF-evolution fitness function.

Usage:
    python diagnostic_true_oracle_hv.py --logs_dir logs
"""

import argparse
import json
import pathlib

import numpy as np
from pymoo.indicators.hv import HV

OBJECTIVE_DIRECTIONS = ["max", "max", "min"]  # Tm, kD, viscosity


def to_allmax(Y: np.ndarray) -> np.ndarray:
    """Flip 'min' objectives so every column is higher-is-better."""
    Y = Y.copy()
    for j, d in enumerate(OBJECTIVE_DIRECTIONS):
        if d == "min":
            Y[:, j] = -Y[:, j]
    return Y


def hv_of(Y_allmax: np.ndarray, ref_point_allmax: np.ndarray) -> float:
    """pymoo's HV indicator minimises, so negate the all-maximise convention."""
    if len(Y_allmax) == 0:
        return 0.0
    return float(HV(ref_point=-ref_point_allmax)(-Y_allmax))


def nearest_true_y(x_query: np.ndarray, oracle_X: np.ndarray, oracle_Y: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour lookup of a pool candidate's true oracle y, by raw
    Euclidean distance in the unit-cube feature space (bounds() for this
    oracle IS [0,1]^16, so pool_x_norm and oracle_X_raw already live in the
    same coordinate system — no unnormalisation or scaler needed). This is
    a diagnostic-only lookup: it does not consume the oracle's _queried
    budget-tracking set, unlike query_mo(), since we are re-examining
    already-completed campaigns, not spending new experiments.
    """
    dists = np.linalg.norm(oracle_X - x_query, axis=1)
    return oracle_Y[int(np.argmin(dists))]


def greedy_exploit_batch(pool_pred_mu_allmax: np.ndarray, front_allmax: np.ndarray,
                          ref_point_allmax: np.ndarray, batch_size: int) -> list:
    """
    Pure-exploitation counterfactual: greedily pick the batch_size pool
    candidates that add the most GP-PREDICTED hypervolume, one at a time,
    against a front that includes previously greedy-picked candidates too
    (so the batch itself has no redundant near-duplicates) — but using only
    the GP's posterior mean, never its uncertainty. This is deliberately
    the trap flagged earlier: "trust the GP posterior mean" is the simplest
    concrete pure-exploitation baseline.
    """
    remaining = list(range(len(pool_pred_mu_allmax)))
    chosen = []
    current_front = front_allmax.copy()
    current_hv = hv_of(current_front, ref_point_allmax)
    for _ in range(min(batch_size, len(remaining))):
        best_idx, best_gain = None, -np.inf
        for i in remaining:
            trial_front = np.vstack([current_front, pool_pred_mu_allmax[i:i+1]])
            gain = hv_of(trial_front, ref_point_allmax) - current_hv
            if gain > best_gain:
                best_gain, best_idx = gain, i
        chosen.append(best_idx)
        remaining.remove(best_idx)
        current_front = np.vstack([current_front, pool_pred_mu_allmax[best_idx:best_idx+1]])
        current_hv = hv_of(current_front, ref_point_allmax)
    return chosen


def analyse_seed_log(log: dict) -> list:
    """Returns a list of per-step dicts: {actual_gain, exploit_gain, ...}."""
    oracle_X = np.array(log["oracle_X_raw"])
    oracle_Y = np.array(log["oracle_Y_raw"])
    Y_all_allmax = to_allmax(oracle_Y)
    ref_point_allmax = (Y_all_allmax.min(axis=0) -
                         0.1 * (Y_all_allmax.max(axis=0) - Y_all_allmax.min(axis=0) + 1e-9))

    Y_running = np.array(log["Y_init"])
    batch_size = log["batch_size"]
    rows = []

    for step in log["decisions"]:
        if "pool_x_norm" not in step or "pool_pred_mu" not in step:
            # Fallback path was taken this batch (see strategy_mo_egbo_novelty's
            # except branch) — no pool was logged, skip, can't diagnose it.
            continue

        front_allmax = to_allmax(Y_running.copy())
        front_hv = hv_of(front_allmax, ref_point_allmax)

        pool_x = np.array(step["pool_x_norm"])
        pool_pred_mu_allmax = np.array(step["pool_pred_mu"])  # already all-max convention

        # Actual EGBO-novelty pick, true oracle y (already logged directly).
        actual_true_y = np.array(step["picked_y"])
        actual_gain = hv_of(np.vstack([front_allmax, to_allmax(actual_true_y)]),
                             ref_point_allmax) - front_hv

        # Pure-exploitation counterfactual pick (GP-predicted mu only),
        # then look up ITS true oracle y for a fair ground-truth comparison.
        exploit_idx = greedy_exploit_batch(pool_pred_mu_allmax, front_allmax,
                                            ref_point_allmax, batch_size)
        exploit_true_y = np.array([nearest_true_y(pool_x[i], oracle_X, oracle_Y)
                                    for i in exploit_idx])
        exploit_gain = hv_of(np.vstack([front_allmax, to_allmax(exploit_true_y)]),
                              ref_point_allmax) - front_hv

        rows.append({
            "step": step["step"],
            "actual_true_hv_gain": actual_gain,
            "exploit_true_hv_gain": exploit_gain,
            "egbo_wins": actual_gain > exploit_gain,
        })

        Y_running = np.vstack([Y_running, actual_true_y])

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
            for row in analyse_seed_log(log):
                row["condition"] = log["condition"]
                row["seed"] = log["seed"]
                all_rows.append(row)

    if not all_rows:
        print("No diagnosable steps found (check that run_mock_subset.py ran "
              "and pool fields were logged — no fallback-only runs).")
        return

    n = len(all_rows)
    n_win = sum(r["egbo_wins"] for r in all_rows)
    mean_actual = np.mean([r["actual_true_hv_gain"] for r in all_rows])
    mean_exploit = np.mean([r["exploit_true_hv_gain"] for r in all_rows])

    print(f"Steps analysed: {n}")
    print(f"EGBO-novelty (actual pick) beats pure-exploitation on true-oracle HV: "
          f"{n_win}/{n} steps ({100*n_win/n:.1f}%)")
    print(f"Mean true HV gain — actual: {mean_actual:.2f}   exploit-only: {mean_exploit:.2f}")
    print()
    if n_win / n > 0.5 and mean_actual > mean_exploit:
        print("=> MODE A (predicted-vs-actual gap). Proceed to build and tune "
              "the exploration-credit term against this true-oracle-HV signal.")
    else:
        print("=> MODE B (single-step regret is fundamentally wrong, even with "
              "ground truth). Do not patch with an exploration-credit term — "
              "fall back to full counterfactual replay (2b) for AF-evolution "
              "fitness.")

    by_cond = {}
    for r in all_rows:
        by_cond.setdefault(r["condition"], []).append(r["egbo_wins"])
    print("\nBy condition:")
    for cond, wins in by_cond.items():
        print(f"  {cond:<18} {sum(wins)}/{len(wins)} steps EGBO-novelty wins "
              f"({100*sum(wins)/len(wins):.1f}%)")

    # Persist a summary — this script previously only printed, which meant
    # make_figures.py (or anything else downstream) had no way to recover
    # these numbers without a full rerun, and no way at all to recover a
    # PAST run's numbers once overwritten (exactly what happened to the
    # pre-fix evolution run's history.json).
    summary = {
        "n_steps": n, "n_win": n_win, "win_rate": n_win / n,
        "mean_actual_true_hv_gain": float(mean_actual),
        "mean_exploit_true_hv_gain": float(mean_exploit),
        "by_condition": {cond: {"n_win": sum(wins), "n": len(wins),
                                 "win_rate": sum(wins) / len(wins)}
                          for cond, wins in by_cond.items()},
    }
    out_path = pathlib.Path(args.logs_dir).parent / "diagnostic_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
