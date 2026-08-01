def score_pool(context):
    """Exploitation with adaptive uncertainty and novelty: predicted means scaled by progress, plus normalized uncertainty and distance from observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Adaptive UCB weight: more exploration early, more exploitation later
        ucb_weight = 0.3 + 0.7 * progress
        sigma_norm = sum(gp[name]["std"] for name in names) / len(names)
        # Novelty term: inverse distance to nearest observed point
        if X_obs.size > 0:
            dists = np.linalg.norm(X_obs - cand["x"], axis=1)
            novelty = 1.0 / (np.min(dists) + 1e-8)
        else:
            novelty = 1.0
        scores.append(mu_sum + ucb_weight * sigma_norm + 0.1 * novelty)
    return scores