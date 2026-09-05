def score_pool(context):
    """Dynamically balance acquisition value with uncertainty and novelty using progress-aware scaling and inverse distance to pareto front."""
    
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

    # Novelty reward based on inverse distance to the nearest point on Pareto front
    pareto_front = context["pareto_front"]
    
    nov_rewards = []
        
    if len(pareto_front) > 0:
        for cand in context["pool"]:
            x_cand = cand["x"] 
            # Compute squared distances from candidate features to all points in the Pareto front
            dists_sq = np.sum((pareto_front - x_cand)**2, axis=1)
            
            min_dist_sq = np.min(dists_sq) 
            
            if min_dist_sq > 0:
                nov_rewards.append(1. / (min_dist_sq + 1e-8))  
            else: 
                # If candidate is exactly on the front
                nov_rewards.append(float('inf'))  

    else:
        nov_rewards = [0.0] * len(context["pool"])

    final_scores = acq_values + np.array(unc_scores) - np.array(nov_rewards)
    
    return list(final_scores)