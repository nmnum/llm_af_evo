def score_pool(context):
    """Incorporate uncertainty-aware acquisition scores with progress-adaptive exploitation and novelty blending."""
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    campaign_progress = context["campaign"]["progress"]

    # Base normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])

    # Uncertainty bonus with progress-dependent weight
    ucb_weight = 0.5 * (1 - campaign_progress)
    
    unc_bonus = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names) 
        
        # Normalize uncertainty bonus by number of objectives
        unc_bonus.append(ucb_weight * (sigma_norm_sum / len(names)))

    # Novelty score based on inverse distance to nearest observed point  
    X_obs = context["X_obs"]
    
    nov_scores = []
    if len(X_obs) > 0:
        
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            # Use inverse of distance as novelty score (higher is better)
            min_dist_sq = np.min(dists) 
            
            if min_dist_sq == 0:
                nov_score = float('-inf')   # Avoid division by zero
            else:  
                nov_score = -np.log(min_dist_sq + 1e-8)

            nov_scores.append(nov_score)
    else:
       nov_scores = [float("-inf")] * len(context["pool"])
        
    # Blend acquisition, uncertainty and novelty scores 
    final_scores = acq_scores + np.array(unc_bonus) + (0.3 * campaign_progress * np.array(nov_scores))
    
    return list(final_scores)