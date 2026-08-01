def score_pool(context):
    """Exploitation with dynamic uncertainty weighting and novelty bonus, adapted from UCB and EGBO-novelty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Dynamic UCB weight: higher early, decreases as progress increases
        beta = 2.0 * (1.0 - progress)
        ucb_score = mu_sum + beta * sigma_norm
        # Novelty term: inverse of distance to nearest observed point
        x = cand["x"]
        if len(context["X_obs"]) == 0:
            novelty = 1.0
        else:
            distances = np.linalg.norm(context["X_obs"] - x, axis=1)
            min_distance = np.min(distances)
            novelty = 1.0 / (1e-8 + min_distance)
        scores.append(ucb_score + 0.5 * novelty)
    return scores