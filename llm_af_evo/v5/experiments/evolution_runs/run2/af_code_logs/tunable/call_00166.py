def score_pool(context):
    """Score candidates by balancing acquisition value with entropy-based uncertainty and progress-aware exploration weight."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
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

    # Normalize distances to [0, 1] using max distance in current observations  
    if len(X_obs) > 0:
        dist_range = np.max(min_distances)
        if dist_range == 0: 
            normalized_dists = min_distances
        else:
            normalized_dists = min_distances / (dist_range + 1e-8)
    else:
        # If nothing observed, set all distances to zero  
        normalized_dists = np.zeros_like(min_distances)

    
    campaign_progress = context["campaign"]["progress"]
        
    exploration_weight = max(0.3, 0.7 * (1 - campaign_progress))

    final_scores = []
    for i in range(len(acq_scores)):
            
        # Novelty term based on inverse of normalized distance  
        novelty_term = (1 - normalized_dists[i]) 
        
        uncertainty_term = np.mean([cand["gp_posterior"][name]["std"] 
                                    for name in names])
        
        entropy_uncertainty = 0.5 * np.log(2*np.pi*np.e*(uncertainty_term**2))
          
        combined_score = exploration_weight * acq_scores[i] + \
                         (1 - exploration_weight) * novelty_term
        
        final_scores.append(combined_score)
        
    return final_scores