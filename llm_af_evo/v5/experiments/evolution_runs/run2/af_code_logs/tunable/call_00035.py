def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Add UCB-style uncertainty bonus
    unc_bonus_weight = 0.5 
    front_range = context["pareto_front_range"]
    sigma_norm_sum = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        norm_std = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        sigma_norm_sum.append(norm_std)

    # Compute novelty bonus based on distance to nearest observed point
    nov_bonus_weight = 0.1  
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        cand_xs = np.array([cand['x'] for cand in context["pool"]])
        
        # Vectorized nearest neighbor search
        diffs = np.sum((X_obs[:, None, :] - cand_xs[None, :, :]) ** 2, axis=2)
        min_distances = np.min(diffs, axis=0)  
        obs_dists = min_distances / len(cand_xs[0])
    else:
        obs_dists = [0.0] * len(context["pool"])
        
    # Combine all components with weights
    final_scores = acq_scores + unc_bonus_weight * np.array(sigma_norm_sum) - nov_bonus_weight * np.array(obs_dists)
    
    return list(final_scores)