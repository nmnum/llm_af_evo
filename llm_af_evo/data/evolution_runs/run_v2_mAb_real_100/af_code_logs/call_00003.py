def score_pool(context):
    """Combine UCB-style exploration credit with explicit novelty term."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_score = mu_sum + 1.0 * sigma_norm
        # Novelty term: inverse of distance to nearest observed point
        cand_x = cand["x"]
        if X_obs.size == 0:
            novelty = 1.0
        else:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            novelty = 1.0 / (1e-8 + min_distance)  # avoid division by zero
        scores.append(ucb_score + 0.5 * novelty)
    return scores