def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted novelty and dynamic exploitation-exploitation balance."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-adaptive UCB bonus
    progress = campaign["progress"]
    ucb_weight = 0.5 * (1 - progress**2) 
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(ucb_weight * sigma_sum)

    # Novelty based on distance to Pareto frontier
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        Y_obs = context["Y_obs"]
        
        nov_rewards = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            
            # Compute distances from candidate's predicted objectives to the Pareto front
            pred_obj = np.array([cand['gp_posterior'][name]["mean"] for name in names])
            
            dists_to_front = np.sum((context["pareto_front"] - pred_obj)**2, axis=1)
            min_dist_to_front = np.min(dists_to_front) 
            
            # Normalize by the range of observed objectives
            front_range_vals = [front_range[name] for name in names]
            
            norm_min_dist = min_dist_to_front / (np.sum(np.array(front_range_vals)**2))
                
            nov_rewards.append(norm_min_dist)
    else:
        nov_rewards = [0.0] * len(context["pool"])

    # Combine terms: exploitation via acquisition, uncertainty bonus and novelty
    final_scores = acq_values + np.array(unc_scores) - 1.5*np.array(nov_rewards)

    return list(final_scores)