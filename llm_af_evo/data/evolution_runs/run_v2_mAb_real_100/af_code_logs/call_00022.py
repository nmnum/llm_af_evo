def score_pool(context):
    """Exploitation with adaptive uncertainty and novelty: predicted means scaled by progress, plus normalized uncertainty and distance from observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Adaptive UCB weight: higher early, decreasing later
        ucb_weight = 1.0 - 0.5 * progress  # 1.0 to 0.5
        sigma_norm = sum(gp[name]["std"] / context["pareto_front_range"][name] for name in names)
        # Novelty term: inverse distance to nearest observed point
        if len(X_obs) > 0:
            dist = np.min(np.linalg.norm(X_obs - cand["x"], axis=1))
            novelty = 1.0 / (1.0 + dist)
        else:
            novelty = 1.0
        scores.append(mu_sum * ucb_weight + sigma_norm + novelty)
    return scores