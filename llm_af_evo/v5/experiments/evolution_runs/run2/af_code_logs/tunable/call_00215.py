def score_pool(context):
    """Leverage uncertainty-aware hypervolume improvement estimation with progressive exploration emphasis and dynamic novelty scaling."""
    names = context["objective_names"]
    
    # Use the normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from each candidate to the nearest observed point
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        min_distances_squared = np.ones(len(context["pool"])) * float('inf')
    else:
        candidates_x = np.array([cand['x'] for cand in context["pool"]])
        
        # Compute pairwise squared Euclidean distances
        expanded_X_obs = X_obs[:, None, :] 
        expanded_candidates = candidates_x[None, :, :]
        diff_sq = (expanded_X_obs - expanded_candidates) ** 2  
        sum_diff_sq = np.sum(diff_sq, axis=2)
        min_distances_squared = np.min(sum_diff_sq, axis=0)

    # Normalize distances to [0,1] using the maximum distance in current observations
    if len(X_obs) > 0:
        dist_range = np.max(min_distances_squared)**.5 
        normalized_dists = (min_distances_squared**.5 / max(dist_range, 1e-8))
    else:  
        # If nothing observed yet set all distances to zero for novelty calculation
        normalized_dists = np.zeros(len(context["pool"]))

    campaign_progress = context["campaign"]["progress"]
    
    # Progress-aware exploration weight with sigmoidal decay from high (0.7) at start 
    exp_weight = 1 - 0.3 * (np.tanh(2*(campaign_progress-0.5)) + 1)/2
    
    front_range = context["pareto_front_range"] 
    
    unc_scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        
        # Compute uncertainty as sum of normalized standard deviations
        sigma_norm_sum = np.sum(gp_posterior[name]["std"]/front_range[name] 
                                for name in names)
                
        # Scale the UCB bonus by progress to encourage earlier exploration  
        ucb_bonus_scaled = exp_weight * 2.0 * (1 - campaign_progress) ** .5
        
        unc_scores.append(ucb_bonus_scaled * sigma_norm_sum)

    final_scores = []
    
    for i, _ in enumerate(context["pool"]):
        
        # Combine acquisition score with uncertainty bonus and novelty term  
        combined_score = acq_scores[i] + \
                         unc_scores[i] - \
                         (1 - normalized_dists[i]) ** 2

        final_scores.append(combined_score)
    
    return final_scores