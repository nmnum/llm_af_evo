def score_pool(context):
    """Exploitation with adaptive uncertainty weighting and novelty bonus: balance predicted means, uncertainty, and distance from observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive UCB weight: more uncertainty early, less later
        ucb_weight = 1.0 - 0.5 * progress
        # Novelty bonus: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty = 1.0
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty)
    return scores