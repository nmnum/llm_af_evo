def score_pool(context):
    """Blend acquisition value with progress-adaptive uncertainty and stagnation-informed novelty to balance exploration-exploitation."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-aware UCB scaling
    progress = campaign["progress"]
    ucb_weight = 0.5 * (1 - progress**2) 
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(ucb_weight * sigma_sum)

    # Novelty reward based on stagnation and distance to observations
    stagnant_batches = campaign["stagnant_batches"]
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        nov_rewards = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            
            # Normalize and weight by stagnation
            norm_min_dist = min_dist / (len(x_cand) * max(1., stagnant_batches))
            nov_rewards.append(norm_min_dist)
    else:
        nov_rewards = [0.0] * len(context["pool"])

    final_scores = acq_values + np.array(unc_scores) - 0.5 * np.array(nov_rewards)
    
    return list(final_scores)