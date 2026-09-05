def score_pool(context):
    """Integrate acquisition value with a progress-adaptive uncertainty bonus and inverse novelty to encourage diverse exploration while maintaining exploitation momentum."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from each candidate to the nearest observed point
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        min_distances = np.ones(len(context["pool"])) * float('inf')
    else:
        candidates_x = np.array([cand['x'] for cand in context["pool"]])
        
        # Compute pairwise squared Euclidean distances
        expanded_X_obs = X_obs[:, None, :] 
        expanded_candidates = candidates_x[None, :, :]
        
        diff_sq = (expanded_X_obs - expanded_candidates) ** 2  
        sum_diff_sq = np.sum(diff_sq, axis=2)
        
        min_distances_squared = np.min(sum_diff_sq, axis=0)
        # Avoid division by zero
        min_distances = np.sqrt(min_distances_squared + 1e-8)

    # Normalize distances to [0, 1] using max distance in current observations (or a small default if no obs yet)  
    if len(X_obs) > 0:
        dist_range = np.max(min_distances)
        if dist_range == 0: 
            normalized_dists = min_distances
        else:
            normalized_dists = min_distances / dist_range
    else:
        # If nothing observed, set all distances to zero (or some default like small value)  
        normalized_dists = np.zeros_like(min_distances)
    
    campaign_progress = context["campaign"]["progress"]
        
    # Progress-adaptive exploration weight: start high, decay towards later stages 
    exploration_weight = 0.7 * (1 - campaign_progress)

    front_range = context["pareto_front_range"] 
    
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        # Scale uncertainty bonus by exploration weight
        unc_score = exploration_weight * sigma_norm_sum  
        unc_scores.append(unc_score)

    final_scores = []
    
    for i, _ in enumerate(acq_scores):
            
        novelty_term = (1 - normalized_dists[i]) 
          
        combined_score = acq_scores[i] + \
                        _unc_scores[i] + \
                         0.5 * novelty_term
            
        final_scores.append(combined_score)
        
    return final_scores