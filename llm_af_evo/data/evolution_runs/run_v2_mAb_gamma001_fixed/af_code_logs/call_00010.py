def score_pool(context):
    """Exploitation with progress-adaptive uncertainty bonus and novelty reward."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # UCB-style uncertainty bonus, adapting weight by progress
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_weight = 2.0 * (1 - context["campaign"]["progress"])
        ucb_score = mu_sum + ucb_weight * sigma_norm
        
        # Add novelty bonus: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novel_bonus = 1.0 / (np.min(distances) + 1e-8)
        else:
            novel_bonus = 1.0
            
        scores.append(ucb_score * (1 + 0.5 * novel_bonus))
    
    return scores