def score_pool(context):
    """Adaptive UCB with novelty bonus, balancing exploitation and uncertainty based on progress."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Dynamically adjust the weight between mean and std based on progression
    w_exploit = 0.2 + 0.8 * progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names) 
        
        # UCB-style score: weighted mean and normalized std
        ucb_score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm
        
        # Novelty term based on minimum distance to observed points
        novelty = np.linalg.norm(X_obs - cand["x"], axis=1).min()
        
        scores.append(ucb_score + 2.7638 * novelty)
    
    return scores