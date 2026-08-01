def score_pool(context):
    """Exploitation with adaptive uncertainty and novelty: predicted means scaled by progress, plus normalized uncertainty and distance from observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Adaptive UCB weight: start low, increase towards the end
        ucb_weight = 0.3 + 0.7 * progress
        sigma_norm = sum(gp[name]["std"] for name in names) / (1e-8 + max(gp[name]["mean"] for name in names))
        # Novelty term: inverse of distance to nearest observed point
        if len(X_obs) == 0:
            novelty = 0.0
        else:
            distances = np.sum((X_obs - cand["x"]) ** 2, axis=1)
            novelty = 1.0 / (1e-8 + np.min(distances))
        scores.append(mu_sum + ucb_weight * sigma_norm + 0.1 * novelty)
    return scores