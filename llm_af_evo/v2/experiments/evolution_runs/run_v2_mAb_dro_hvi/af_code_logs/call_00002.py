def score_pool(context):
    """Combine UCB-style exploration with novelty term based on distance to nearest observed point."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    pool = context["pool"]
    
    scores = []
    for cand in pool:
        gp = cand["gp_posterior"]
        
        # UCB-style exploration: mean sum plus uncertainty bonus
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_score = mu_sum + 1.0 * sigma_norm
        
        # Novelty term: inverse of distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) == 0:
            novelty = 1e6  # No observations, so very novel
        else:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            novelty = 1.0 / (min_distance + 1e-9)  # Avoid division by zero
            
        scores.append(ucb_score + 0.5 * novelty)

    return scores