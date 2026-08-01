"""
fitness_common.py — shared HV/fitness helpers for llm_af_evo, used by both
the pre-test scripts (diagnostic_true_oracle_hv.py, sweep_lambda.py) and the
L1 evolution pipeline (generate_training_set.py, evolve_af.py, validate_l1.py).

Fitness definition (as decided in the L1 build-out): for a candidate AF and
a logged step, compare the AF's own GP-only exploration-credit fitness
(HV_pred_improvement + LAMBDA_STAR * sigma_norm — no oracle, deployable on
real data) against EGBO-novelty's actual pick's fitness computed the SAME
way. A step is a "win" if the evolved AF's fitness exceeds EGBO-novelty's.
This deliberately reuses sweep_lambda.py's validated lambda*=10000 form
rather than switching to a true-oracle metric for training, so the fitness
used to evolve/select AFs is identical in kind to what would be computable
online during a real (non-synthetic) campaign.
"""

import numpy as np
from pymoo.indicators.hv import HV

OBJECTIVE_DIRECTIONS = ["max", "max", "min"]  # Tm, kD, viscosity
LAMBDA_STAR = 10000  # from sweep_lambda.py's validated plateau (win_rate=0.678)


def to_allmax(Y: np.ndarray, directions=None) -> np.ndarray:
    """
    Flip 'min' objectives so every column is higher-is-better.

    directions defaults to the module-level OBJECTIVE_DIRECTIONS (Tm, kD,
    viscosity) for backward compatibility with every existing excipient-
    only call site — pass an oracle's own .objective_directions() when
    working with a different oracle (e.g. all-max coatings objectives),
    same reasoning as excipient_campaign_mo.pareto_front_of's directions
    parameter.
    """
    if directions is None:
        directions = OBJECTIVE_DIRECTIONS
    Y = Y.copy()
    for j, d in enumerate(directions):
        if d == "min":
            Y[:, j] = -Y[:, j]
    return Y


def hv_of(Y_allmax: np.ndarray, ref_point_allmax: np.ndarray) -> float:
    """pymoo's HV indicator minimises, so negate the all-maximise convention."""
    if len(Y_allmax) == 0:
        return 0.0
    return float(HV(ref_point=-ref_point_allmax)(-Y_allmax))


def nearest_true_y(x_query: np.ndarray, oracle_X: np.ndarray, oracle_Y: np.ndarray) -> np.ndarray:
    """Diagnostic-only nearest-neighbour lookup — does not touch _queried."""
    dists = np.linalg.norm(oracle_X - x_query, axis=1)
    return oracle_Y[int(np.argmin(dists))]


def sigma_norm_of_set(sigma_set: np.ndarray, range_j: np.ndarray) -> float:
    """sum over candidates in the set of (sum over objectives of sigma_j/range_j)."""
    return float(np.sum(sigma_set / range_j))


def greedy_exploit_batch(pool_pred_mu_allmax: np.ndarray, front_allmax: np.ndarray,
                          ref_point_allmax: np.ndarray, batch_size: int) -> list:
    """Pure-exploitation counterfactual (GP posterior mean only, no uncertainty)."""
    remaining = list(range(len(pool_pred_mu_allmax)))
    chosen = []
    current_front = front_allmax.copy()
    current_hv = hv_of(current_front, ref_point_allmax)
    for _ in range(min(batch_size, len(remaining))):
        best_idx, best_gain = None, -np.inf
        for i in remaining:
            trial_front = np.vstack([current_front, pool_pred_mu_allmax[i:i + 1]])
            gain = hv_of(trial_front, ref_point_allmax) - current_hv
            if gain > best_gain:
                best_gain, best_idx = gain, i
        chosen.append(best_idx)
        remaining.remove(best_idx)
        current_front = np.vstack([current_front, pool_pred_mu_allmax[best_idx:best_idx + 1]])
        current_hv = hv_of(current_front, ref_point_allmax)
    return chosen


def af_fitness_of_set(idx_set, pool_mu_allmax, pool_sigma, front_allmax,
                       ref_point_allmax, range_j, lam: float = LAMBDA_STAR) -> float:
    """
    HV_pred_improvement(set) + lam * sigma_norm(set) — the GP-only,
    deployable exploration-credit fitness validated in sweep_lambda.py.

    NOT used as L1's training/evolution fitness (see evolve_af.py) — at
    lambda*=10000 this is dominated by the sigma_norm term almost
    regardless of mu, so scoring CANDIDATE AFs by it during evolution is
    gameable: an AF that maximises sigma_norm alone, ignoring mu entirely,
    scores near-optimally without needing to predict anything well. This
    was discovered empirically (run1's evolved population converged to
    exactly that degenerate form, win_rate=1.0 from generation 0). Still
    correct and still used for its original purposes: the pre-test's
    diagnostic_true_oracle_hv.py / sweep_lambda.py validation, and as a
    reference for what a real, oracle-free ONLINE campaign could compute
    (relevant for L2, not for L1's offline training signal).
    """
    mu_set = pool_mu_allmax[idx_set]
    front_hv = hv_of(front_allmax, ref_point_allmax)
    hv_pred_gain = hv_of(np.vstack([front_allmax, mu_set]), ref_point_allmax) - front_hv
    s_norm = sigma_norm_of_set(pool_sigma[idx_set], range_j)
    return hv_pred_gain + lam * s_norm


def true_hv_gain_of_pick(idx_set, pool_x, oracle_X, oracle_Y,
                          front_allmax, ref_point_allmax) -> float:
    """
    True-oracle HV gain of a candidate set — the L1 training/evolution
    fitness target. Legitimate here despite using ground truth: L1 trains
    entirely offline on synthetic campaigns, and only the FINAL evolved
    score_pool's own decision rule needs to be oracle-free for deployment
    (which it always is — score_pool only ever sees mu/sigma/x, never
    oracle_X/oracle_Y). Using true outcomes to SELECT among candidate AFs
    during training is not a deployment-time leak, and unlike the GP-only
    proxy above, actually maximising this requires genuinely good picks,
    not just high uncertainty.
    """
    true_y = np.array([nearest_true_y(pool_x[i], oracle_X, oracle_Y) for i in idx_set])
    front_hv = hv_of(front_allmax, ref_point_allmax)
    return hv_of(np.vstack([front_allmax, to_allmax(true_y)]), ref_point_allmax) - front_hv


def step_context(log: dict, Y_running: np.ndarray, X_running: np.ndarray) -> dict:
    """
    Assemble the AF-interface context for one training/eval step, given the
    running (already-observed) state accumulated so far in this campaign.
    front_allmax/ref_point_allmax/range_j are exactly what a real AF would
    have available online — no oracle ground truth involved.
    """
    oracle_Y = np.array(log["oracle_Y_raw"])
    Y_all_allmax = to_allmax(oracle_Y)
    ref_point_allmax = (Y_all_allmax.min(axis=0) -
                         0.1 * (Y_all_allmax.max(axis=0) - Y_all_allmax.min(axis=0) + 1e-9))
    front_allmax = to_allmax(Y_running.copy())
    range_j = np.maximum(np.ptp(Y_running, axis=0), 1e-6)
    return {
        "front_allmax": front_allmax,
        "ref_point_allmax": ref_point_allmax,
        "range_j": range_j,
        "X_obs": X_running.copy(),
    }
