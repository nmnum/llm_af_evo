"""
novelty_selection.py — Novelty-aware batch selection for multi-objective EGBO.

Implements the novelty-aware selection strategy from Aqeeli et al. (2026):
candidates are selected sequentially using a hybrid score that combines
predicted acquisition merit with novelty (distance from already-selected
points in normalised input space).

score = w_acq * acquisition_merit_normalised + w_nov * novelty_normalised

This prevents redundant candidates within a batch — the key failure mode
of standard EGBO where high-scoring candidates cluster in the same region
of design space, wasting experiment slots.

Usage:
    from novelty_selection import novelty_aware_select

    selected_idx = novelty_aware_select(
        candidates_n,      # (N, d) normalised candidates in [0,1]^d
        acq_scores,        # (N,) acquisition values (higher = better)
        n_select=5,        # batch size
        w_acq=0.9,         # acquisition weight
        w_nov=0.1,         # novelty weight (validated via Phase 1 sweep; 0.3 is
                            # the Aqeeli et al. default but is too aggressive at
                            # budget=30 for this domain)
    )
"""

import numpy as np


def novelty_aware_select(
    candidates_n: np.ndarray,
    acq_scores: np.ndarray,
    n_select: int,
    w_acq: float = 0.9,
    w_nov: float = 0.1,
    X_obs_n: np.ndarray = None,
) -> np.ndarray:
    """
    Sequentially select n_select candidates balancing acquisition merit
    and novelty (distance from already-selected and already-observed points).

    Parameters
    ----------
    candidates_n : (N, d) array
        Candidate points in normalised [0,1]^d space.
    acq_scores : (N,) array
        Acquisition function values for each candidate (higher = better).
        These can be qLogNEHVI values, UCB, or any merit score.
    n_select : int
        Number of candidates to select (batch size).
    w_acq : float
        Weight on acquisition merit (0-1). Default 0.9 (validated via Phase 1
        sweep for this domain; Aqeeli et al. use 0.7).
    w_nov : float
        Weight on novelty (0-1). Default 0.1 (validated via Phase 1 sweep;
        Aqeeli et al.'s 0.3 was found too aggressive at budget=30 here).
    X_obs_n : (M, d) array or None
        Already-observed points in normalised space. Novelty is computed
        against both selected candidates AND existing observations, so we
        don't re-sample near known points.

    Returns
    -------
    selected_idx : (n_select,) int array
        Indices into candidates_n of the selected batch.
    """
    N = len(candidates_n)
    n_select = min(n_select, N)

    # Normalise acquisition scores to [0, 1]
    a_min, a_max = float(acq_scores.min()), float(acq_scores.max())
    if a_max - a_min < 1e-12:
        acq_norm = np.ones(N) * 0.5
    else:
        acq_norm = (acq_scores - a_min) / (a_max - a_min)

    # Track selected and compute novelty incrementally
    selected = []
    available = list(range(N))

    # Precompute distances to existing observations if provided
    if X_obs_n is not None and len(X_obs_n) > 0:
        dist_to_obs = np.array([
            np.min(np.linalg.norm(X_obs_n - candidates_n[i], axis=1))
            for i in range(N)
        ])
    else:
        dist_to_obs = np.full(N, np.inf)  # no observations → novelty from obs is max

    for _ in range(n_select):
        if not available:
            break

        # Compute novelty for each available candidate:
        # min distance to already-selected points, and to existing observations
        novelties = np.full(N, -np.inf)
        for i in available:
            if selected:
                dist_to_sel = np.min([
                    np.linalg.norm(candidates_n[i] - candidates_n[j])
                    for j in selected
                ])
            else:
                dist_to_sel = np.inf
            # Novelty = max of (dist to selected, dist to observations)
            # If no observations, dist_to_obs is inf so novelty = dist_to_sel
            novelties[i] = max(dist_to_sel, dist_to_obs[i])

        # Normalise novelties among available candidates
        avail_nov = novelties[available]
        # Handle case where all novelties are inf (first pick, no obs)
        finite_mask = np.isfinite(avail_nov)
        if not finite_mask.any():
            nov_norm_avail = np.ones(len(available)) * 0.5
        else:
            n_min = float(avail_nov[finite_mask].min())
            n_max = float(avail_nov[finite_mask].max())
            if n_max - n_min < 1e-12:
                nov_norm_avail = np.ones(len(available)) * 0.5
            else:
                nov_norm_avail = np.where(
                    finite_mask,
                    (avail_nov - n_min) / (n_max - n_min),
                    1.0  # inf novelty → max normalised novelty
                )

        # Combined score for available candidates
        combined = np.array([
            w_acq * acq_norm[available[k]] + w_nov * nov_norm_avail[k]
            for k in range(len(available))
        ])

        # Pick the best
        best_local = int(np.argmax(combined))
        best_global = available[best_local]
        selected.append(best_global)
        available.remove(best_global)

    return np.array(selected, dtype=int)


def novelty_aware_select_vectorised(
    candidates_n: np.ndarray,
    acq_scores: np.ndarray,
    n_select: int,
    w_acq: float = 0.9,
    w_nov: float = 0.1,
    X_obs_n: np.ndarray = None,
) -> np.ndarray:
    """
    Vectorised version of novelty_aware_select for larger candidate pools.
    Same logic, faster for N > 200.
    """
    N = len(candidates_n)
    n_select = min(n_select, N)

    # Normalise acquisition scores
    a_min, a_max = float(acq_scores.min()), float(acq_scores.max())
    if a_max - a_min < 1e-12:
        acq_norm = np.ones(N) * 0.5
    else:
        acq_norm = (acq_scores - a_min) / (a_max - a_min)

    # Distance to existing observations
    if X_obs_n is not None and len(X_obs_n) > 0:
        # (N, M) distance matrix
        dist_to_obs = np.min(
            np.linalg.norm(
                candidates_n[:, None, :] - X_obs_n[None, :, :],
                axis=2
            ),
            axis=1
        )
    else:
        dist_to_obs = np.full(N, np.inf)

    selected = []
    mask = np.ones(N, dtype=bool)

    for _ in range(n_select):
        if not mask.any():
            break

        avail_idx = np.where(mask)[0]

        # Novelty: distance to nearest selected point
        if selected:
            sel_pts = candidates_n[selected]  # (S, d)
            dist_to_sel = np.min(
                np.linalg.norm(
                    candidates_n[avail_idx][:, None, :] - sel_pts[None, :, :],
                    axis=2
                ),
                axis=1
            )
        else:
            dist_to_sel = np.full(len(avail_idx), np.inf)

        # Combined novelty = max(dist_to_sel, dist_to_obs)
        nov = np.maximum(dist_to_sel, dist_to_obs[avail_idx])

        # Normalise novelty among available
        finite_mask = np.isfinite(nov)
        if not finite_mask.any():
            nov_norm = np.ones(len(avail_idx)) * 0.5
        else:
            n_min = float(nov[finite_mask].min())
            n_max = float(nov[finite_mask].max())
            if n_max - n_min < 1e-12:
                nov_norm = np.ones(len(avail_idx)) * 0.5
            else:
                nov_norm = np.where(
                    finite_mask,
                    (nov - n_min) / (n_max - n_min),
                    1.0
                )

        # Combined score
        combined = w_acq * acq_norm[avail_idx] + w_nov * nov_norm
        best_local = int(np.argmax(combined))
        best_global = avail_idx[best_local]
        selected.append(best_global)
        mask[best_global] = False

    return np.array(selected, dtype=int)
