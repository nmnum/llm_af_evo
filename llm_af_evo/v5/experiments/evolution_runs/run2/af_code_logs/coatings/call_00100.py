def score_pool(context):
    """Blend acquisition value with a progress-adaptive uncertainty bonus and reciprocal novelty penalty to encourage diverse exploration without overfitting to noise."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-aware UCB: reduce uncertainty bonus as we near the end
    progress = campaign["progress"]
    ucb_weight = 0.3 * (1 - progress**2) 
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(ucb_weight * sigma_sum)

    # Novelty reward based on inverse distance to nearest observed point
    X_obs = context["X_obs"]
    nov_rewards = []
        
    if len(X_obs) > 0:
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            
            # Inverse novelty: higher reward when farther from observations
            nov_reward = 1.0 / (min_dist + 1e-8)  
            norm_nov_reward = nov_reward * len(x_cand) 
            nov_rewards.append(norm_nov_reward)
    else:
        nov_rewards = [0.0] * len(context["pool"])

    # Combine acquisition, uncertainty and novelty terms
    final_scores = acq_values + np.array(unc_scores) - np.array(nov_rewards)
    
    return list(final_scores)