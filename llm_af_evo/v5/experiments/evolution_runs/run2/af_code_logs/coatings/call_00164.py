def score_pool(context):
    """Integrate hypervolume acquisition with adaptive entropy-based exploration to favor candidates near Pareto front boundaries."""
    
    names = context["objective_names"]
    campaign = context["campaign"] 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Entropy-inspired uncertainty bonus: scale by how close we are to end of budget
    progress = campaign["progress"]
    entropy_weight = 0.3 * (1 - progress)
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        # Use entropy bonus to encourage exploration near the Pareto frontier
        unc_scores.append(entropy_weight * (1 + np.log(sigma_sum + 1e-8)))

    X_obs = context["X_obs"]
    
    nov_rewards = []
        
    if len(X_obs) > 0:
        Y_obs = context["Y_obs"] 
        # Compute the inverse density of observations around each candidate
        for cand in context["pool"]:
            x_cand = cand["x"] 
            
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            min_dist = np.min(dists) if len(dists) > 0 else float('inf')
        
            # Reward candidates that are in sparsely explored regions (i.e., far from existing points).
            nov_rewards.append(1. / max(min_dist, 1e-8))
    else:
        nov_rewards = [0.] * len(context["pool"])

    
    final_scores = acq_values + np.array(unc_scores) - np.array(nov_rewards)
    
    return list(final_scores)