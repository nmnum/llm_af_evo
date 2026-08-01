def score_pool(context):
    """Exploitation with progress-adaptive uncertainty bonus, balancing between greedy objective sum and UCB-style exploration, using normalized uncertainty and adding a novelty term to encourage diverse batch selection."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: more exploitation early, more exploration later
        weight = 1.0 + 2.0 * (1.0 - progress)
        ucb_score = mu_sum + weight * sigma_norm
        
        # Add novelty bonus: inverse of distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty_bonus = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty_bonus = 1.0
        
        scores.append(ucb_score + 0.5 * novelty_bonus)
    return scores