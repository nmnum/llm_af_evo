def score_pool(context):
    """Score candidates by normalized acquisition value adjusted for novelty and uncertainty."""
    names = context["objective_names"]
    
    # Use precomputed acq_value_norm directly from the pool 
    scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Normalize to [0, 1] range
    min_s, max_s = np.min(scores), np.max(scores)
    if abs(max_s - min_s) < 1e-9:
        norm_scores = np.ones_like(scores)
    else:
        norm_scores = (np.array(scores) - min_s) / (max_s - min_s + 1e-9)

    # Add a novelty term based on distance to nearest observed point
    X_obs = context["X_obs"]
    if len(X_obs) > 0:  
        X_pool = np.stack([cand["x"] for cand in context["pool"]])
        
        # Compute pairwise distances between pool candidates and observations 
        dists_sq = np.sum((X_pool[:, None, :] - X_obs[None, :, :]) ** 2, axis=2)
        min_dists = np.min(dists_sq, axis=1) 
        
        # Invert to get novelty (higher is more novel), normalize
        novelties = 1.0 / (min_dists + 1e-9)  
        max_novelty = np.max(novelties)
        
        if abs(max_novelty) > 1e-9:
            norm_novelties = novelties / max_novelty
        else: 
            norm_novelties = np.zeros_like(novelties)

    # Combine acquisition value and novelty with a dynamic weight based on progress  
    campaign_progress = context["campaign"]["progress"]
    
    w_acq = 0.7 + 0.3 * (1 - campaign_progress)   # More exploitation early, more exploration late
    w_novelty = 1.0 - w_acq
    
    final_scores = []
    for i in range(len(context["pool"])):
        score = w_acq * norm_scores[i] 
        if len(X_obs) > 0:
            score += w_novelty * norm_novelties[i]
        
        # Also incorporate uncertainty (std deviation of objectives)
        total_std_norm = sum(cand["gp_posterior"][name]["std"] for name in names) / len(names)
        front_range = context["pareto_front_range"]
        std_sum_normalized = np.sum([cand["gp_posterior"][n]["std"]/front_range[n] 
                                     for n in names])
        
        # Boost score slightly if candidate has high uncertainty (exploration bonus)
        ucb_bonus = 0.1 * std_sum_normalized / len(names)  
        final_scores.append(score + ucb_bonus)

    return final_scores