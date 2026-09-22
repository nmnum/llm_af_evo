def score_pool(context):
    """Estimate expected improvement relative to the Pareto front using acquisition values and uncertainty-aware regret scoring."""
    names = context["objective_names"]
    
    # Get base scores (acquisition values already represent hypervolume improvements)
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from each candidate to the nearest point on Pareto front
    pareto_front = context["pareto_front"]
    if len(pareto_front) == 0:
        min_distances_to_pf = np.ones(len(context["pool"])) * float('inf')
    else: 
        candidates_x = np.array([cand['x'] for cand in context["pool"]])
        
        # Compute pairwise squared Euclidean distances between each candidate and all front points
        expanded_front = pareto_front[:, None, :]  
        expanded_candidates = candidates_x[None, :, :]
        
        diff_sq = (expanded_frontend - expanded_candidates) ** 2 
        sum_diff_sq = np.sum(diff_sq, axis=2)
        
        min_distances_squared = np.min(sum_diff_sq, axis=0)
        # Avoid division by zero
        min_distances_to_pf = np.sqrt(min_distances_squared + 1e-8)

    # Normalize distances to [0, 1] using the observed range per objective (or a small default if no obs yet)  
    front_range = context["pareto_front_range"]
    
    normalized_dists_to_pf = []
    for i in range(len(context["pool"])):
        dists_per_obj = np.array([min_distances_to_pf[i]] * len(names))  # Assume equal distance per obj
        norm_dist = np.mean(dists_per_obj / list(front_range.values()))
        if not (norm_dist == norm_dist):   # Check for NaN 
            normalized_dists_to_pf.append(0.5)  
        else:
            normalized_dists_to_pf.append(norm_dist)

    campaign_progress = context["campaign"]["progress"]
    
    exploration_weight = 1 - np.clip(campaign_progress, 0., 1.) ** 2

    final_scores = []
    for i in range(len(acq_values)):
        
        # Use inverse of distance to front as an exploitation bonus
        if normalized_dists_to_pf[i] == float('inf') or not (normalized_dists_to_pf[i]==normalized_dists_to_pf[i]):
            exploit_bonus = 0.5  
        else:
            exploit_bonus = np.clip(1 - normalized_dists_to_pf[i], 0., 1.) 

        # Blend acquisition score with exploitation bonus and adapt weight based on progress
        combined_score = exploration_weight * acq_values[i] + \
                         (1 - exploration_weight) * exploit_bonus
            
        final_scores.append(combined_score)
        
    return final_scores