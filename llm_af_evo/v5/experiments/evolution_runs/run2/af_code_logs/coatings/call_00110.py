def score_pool(context):
    """Leverage acquisition value as primary signal and augment with uncertainty-sensitive novelty reward that adapts to exploration progress."""
    
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

    # Adaptive novelty reward based on how much the candidate's features differ from current observations
    X_obs = context["X_obs"]
    
    nov_rewards = []
        
    if len(X_obs) > 0:
        cand_features = np.array([cand['x'] for cand in context["pool"]])
        feature_diffs = np.sum((cand_features[:, None] - X_obs[None, :])**2, axis=2)
        min_dists = np.min(feature_diffs, axis=1) 
        
        # Normalize by the number of observations and stagnation
        norm_min_dist = min_dists / (len(X_obs) * max(1., campaign["stagnant_batches"]))
        
        nov_rewards = 0.5 * norm_min_dist  
    else:
        nov_rewards = np.zeros(len(context["pool"]))

    # Combine acquisition, uncertainty and novelty terms
    final_scores = acq_values + np.array(unc_scores) - np.array(nov_rewards)
    
    return list(final_scores)