def score_pool(context):
    """Exploitation with progressive uncertainty weighting and novelty penalty: higher predicted means, scaled by progress-driven UCB weight, minus distance from observed points."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Scale uncertainty contribution based on progress: less early, more later
        ucb_weight = 0.5 + 0.5 * progress  # 0.5 to 1.0
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Add novelty penalty: candidates closer to observed points get lower scores
        if len(X_obs) > 0:
            distances = np.sqrt(np.sum((X_obs - cand["x"]) ** 2, axis=1))
            min_distance = np.min(distances)
            # Normalize by the range of features to make the penalty scale-invariant
            feature_range = np.max(X_obs, axis=0) - np.min(X_obs, axis=0)
            feature_range = np.where(feature_range == 0, 1.0, feature_range)
            normalized_distance = min_distance / np.mean(feature_range)
            novelty_penalty = 0.1 * normalized_distance
        else:
            novelty_penalty = 0.0
        scores.append(mu_sum + ucb_weight * sigma_sum - novelty_penalty)
    return scores