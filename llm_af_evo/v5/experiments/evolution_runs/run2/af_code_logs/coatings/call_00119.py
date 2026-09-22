def score_pool(context):
    """Blend acquisition value with progress-adaptive uncertainty and stagnation-aware novelty to balance exploration-exploitation."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-adaptively scale the UCB bonus
    progress = campaign["progress"]
    ucb_weight = 0.5 * (1 - progress**2) 
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(ucb_weight * sigma_sum)

    # Compute novelty reward based on minimum distance to observed points
    X_obs = context["X_obs"]
    nov_rewards = []
        
    if len(X_obs) > 0:
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            
            # Normalize by feature dimensionality
            norm_min_dist = min_dist / len(x_cand)  
            nov_rewards.append(norm_min_dist)
    else:
        nov_rewards = [0.0] * len(context["pool"])

    # Combine terms: acquisition + uncertainty bonus - novelty penalty (scaled by stagnation)
    stagnant_batches = campaign["stagnant_batches"]
    final_scores = acq_values + np.array(unc_scores) - np.array(nov_rewards) / max(1., stagnant_batches)
    
    return list(final_scores)