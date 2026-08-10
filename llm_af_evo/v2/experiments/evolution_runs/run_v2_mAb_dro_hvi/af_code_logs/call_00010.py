def score_pool(context):
    """Blend exploitation with uncertainty and novelty to outperform both pure greedy and hypervolume-based methods."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Base scores: exploit mean objectives + uncertainly (UCB-style)
    base_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_score = mu_sum + 1.0 * sigma_norm
        base_scores.append(ucb_score)

    # Add novelty term: inverse distance to nearest observed point  
    obs_x = context["X_obs"]
    scores = []
    
    if len(obs_x) == 0:
        # No observations yet, use just UCB 
        return base_scores
    
    for cand in context["pool"]:
        x = cand["x"]        
        distances_sq = np.sum((obs_x - x)**2, axis=1)
        min_dist_sq = np.min(distances_sq)  
        novelty_score = 1.0 / (min_dist_sq + 1e-8) 
        scores.append(base_scores[len(scores)] * (1.0 + 0.5 * novelty_score))
    
    return scores