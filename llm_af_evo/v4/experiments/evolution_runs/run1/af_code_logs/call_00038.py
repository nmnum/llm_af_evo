def score_pool(context):
    """Combine UCB-style exploration credit with explicit novelty term."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # UCB-style exploration: sum of means plus uncertainty bonus
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_score = mu_sum + 1.0 * sigma_norm
        
        # Novelty term: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.sqrt(np.sum((X_obs - cand_x)**2, axis=1))
            novelty_score = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty_score = 1.0
        
        # Combine UCB and novelty
        score = ucb_score + 0.5 * novelty_score
        scores.append(score)
    
    return scores