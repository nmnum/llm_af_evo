def score_pool(context):
    """Exploitation with uncertainty bonus: sum of means plus a scaled uncertainty term, adjusted by progress and novelty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    progress = context["campaign"]["progress"]
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Adaptive beta: higher early, lower late
        beta = 2.0 * (1.0 - progress)
        
        # Novelty bonus: distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty = np.min(distances)
        else:
            novelty = 1.0
        
        # Combine exploitation, uncertainty, and novelty
        scores.append(mu_sum + beta * sigma_norm + 0.1 * novelty)
    
    return scores