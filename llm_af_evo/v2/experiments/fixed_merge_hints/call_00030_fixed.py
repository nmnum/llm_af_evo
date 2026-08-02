def score_pool(context):
    """Exploitation with uncertainty-aware novelty: prefer high-mean candidates but penalize those near already-scored points in feature space (sign-fixed from run_v2_mAb_gamma001_fixed's call_00030: novelty must increase with distance to the nearest observed point, not decrease -- the original used 1/min_dist_sq, which scored exact duplicates of past observations as maximally "novel")."""
    names = context["objective_names"]

    means = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names)
                      for cand in context["pool"]])

    ifmeans_max, ifmeans_min = means.max(), means.min()
    norm_means = (np.zeros_like(means) if ifmeans_max == ifmeans_min
                  else (means - ifmeans_min) / (ifmeans_max - ifmeans_min))

    X_pool = np.array([cand["x"] for cand in context["pool"]])

    dists_to_observed = []
    for x_cand in X_pool:
        diffs = context['X_obs'] - x_cand
        dist_sq = np.sum(diffs**2, axis=1)
        min_dist_sq = dist_sq.min()
        # FIXED: original computed 1.0 / min_dist_sq (with an inf special
        # case at exact duplicates) -- a quantity that is LARGEST for
        # candidates closest to an existing observation. Novelty should be
        # proportional to distance itself, not its reciprocal.
        dists_to_observed.append(min_dist_sq)

    nov_max, nov_min = max(dists_to_observed), min(dists_to_observed)
    if np.isclose(nov_max, nov_min):
        norm_novelty_scores = np.zeros_like(dists_to_observed)
    else:
        norm_novelty_scores = (np.array(dists_to_observed) - nov_min) / (nov_max - nov_min + 1e-9)

    final_score = norm_means * norm_novelty_scores

    return list(final_score)
