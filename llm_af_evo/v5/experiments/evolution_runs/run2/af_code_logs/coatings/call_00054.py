def score_pool(context):
    """Use acquisition value as primary signal but dynamically adjust uncertainty and novelty weights based on campaign progress to balance exploration-exploitation."""
    names = context["objective_names"]
    
    # Normalize acq values for consistent blending  
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    if len(acq_values) > 1:
        acq_mean, acq_std = np.mean(acq_values), np.std(acq_values)
        normalized_acqs = (acq_values - acq_mean) / max(1e-8, acq_std)
    else: 
        normalized_acqs = acq_values
    
    # Progress-aware weights
    progress = context["campaign"]["progress"]
    
    # Early phase: more uncertainty and novelty bonus  
    unc_weight = 0.3 * (1 - progress) + 0.1 * progress
    nov_weight = 0.2 * (1 - progress) + 0.05 * progress
    
    front_range = context["pareto_front_range"]
    
    # Uncertainty scores based on normalized GP std 
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]  
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(unc_weight * sigma_norm_sum)

    # Novelty scores based on distance to nearest observed point
    nov_scores = []
    
    if len(context["X_obs"]) > 0:
        X_obs = context["X_obs"]
        
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)  
            
            # Normalize by input space dimensionality
            nov_scores.append(-nov_weight * (min_dist / len(x_cand)))
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    final_scores = normalized_acqs + unc_scores + nov_scores
    
    return list(final_scores)