def score_pool(context):
    """Score by predicted objective sum adjusted for uncertainty and penalized based on proximity to existing observations (sign-fixed from run_v2_mAb_gamma001_fixed's call_00016: penalty must grow as distance to the nearest observation shrinks, not the reverse)."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]

    mu_sums = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names)
                        for cand in context["pool"]])

    sigmas_normed = np.array([
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        for cand in context["pool"]
    ])

    ucb_scores = mu_sums - 0.5 * sigmas_normed

    if len(context["X_obs"]) == 0:
        return list(ucb_scores)

    X_pool = np.array([cand["x"] for cand in context["pool"]])

    dists_to_observed = np.min(np.sum((X_pool[:, None, :] - context["X_obs"][None, :, :]) ** 2, axis=2), axis=1)

    min_dist_penalty = 0.5
    max_dist_penalty = 3.0
    max_dist = np.max(dists_to_observed) + 1e-9

    # FIXED: original computed dists_to_observed / max_dist, which is LARGE
    # for far/novel candidates and SMALL for near-duplicates -- backwards for
    # a quantity meant to penalize proximity. Using (max_dist - dist) / max_dist
    # instead makes the penalty large when close to an observation, small
    # when far, matching the stated intent.
    dist_penalties = np.clip((max_dist - dists_to_observed) / max_dist,
                              min_dist_penalty, max_dist_penalty)

    final_scores = ucb_scores - dist_penalties

    return list(final_scores)
