def score_pool(context):
    """Score by predicted objective sum plus a dynamic uncertainty bonus based on progress and novelty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    
    # Dynamic weight for uncertainty based on progress
    w = 1.0 + 2.0 * (1.0 - progress)  # Decrease uncertainty weight as we progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        
        # Add novelty bonus: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.sqrt(np.sum((X_obs - cand_x)**2, axis=1))
            novelty_bonus = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty_bonus = 1.0
            
        scores.append(mu_sum + w * sigma_sum + 0.5 * novelty_bonus)
    
    return scores