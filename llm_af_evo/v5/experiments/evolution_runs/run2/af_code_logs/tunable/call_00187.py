def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and reciprocal exploration depth for diverse yet focused search."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-adaptive UCB bonus: decrease uncertainty weight as we approach the end
    progress = campaign["progress"]
    ucb_weight = 0.5 * (1 - progress**2) 
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(ucb_weight * sigma_sum)

    # Reciprocal exploration depth: reward points that are farther from existing observations
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        nov_rewards = []
        
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            # Use the inverse of squared distance as novelty reward (higher is better)  
            min_dist_sq = np.min(dists**2)
            
            if min_dist_sq == 0:
                nov_rewards.append(0.0) 
            else:   
                norm_min_dist = 1 / min_dist_sq
                # Normalize by feature dimensionality to avoid bias toward low-D spaces
                normalized_nov = norm_min_dist / len(x_cand)
                nov_rewards.append(normalized_nov)

    else:
        nov_rewards = [0.0] * len(context["pool"])

    final_scores = acq_values + np.array(unc_scores) - 1.5 * np.array(nov_rewards)
    
    return list(final_scores)