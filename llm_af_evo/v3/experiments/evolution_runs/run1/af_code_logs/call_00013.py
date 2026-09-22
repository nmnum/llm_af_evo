def score_pool(context):
    """Weighted combination of UCB-style uncertainty and novelty, with adaptive exploration-exploitation balance."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Adaptive weight for exploitation vs uncertainty (exploit more as we advance)
    w_exploit = 0.3 + 0.7 * progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # UCB-style score: exploit mean, explore std
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names) 
        ucb_score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm
        
        # Novelty bonus: distance to nearest observed point
        novelty = np.linalg.norm(X_obs - cand["x"], axis=1).min()
        
        scores.append(ucb_score + 2.7638 * novelty)
    
    return scores