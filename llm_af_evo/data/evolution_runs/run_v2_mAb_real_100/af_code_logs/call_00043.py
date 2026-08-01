def score_pool(context):
    """Exploitation with adaptive uncertainty weighting and novelty bonus: balances mean prediction, uncertainty, and distance from observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive UCB weight: higher early, decreases towards end
        ucb_weight = 1.0 - 0.5 * progress
        # Novelty bonus: inverse distance to nearest observed point
        if len(X_obs) > 0:
            distances = np.sqrt(np.sum((X_obs - cand["x"]) ** 2, axis=1))
            novelty = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty = 1.0
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.1 * novelty)
    return scores