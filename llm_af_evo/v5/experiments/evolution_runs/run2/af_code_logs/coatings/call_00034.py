def score_pool(context):
    """Blend acquisition value with uncertainty-aware and novelty-informed scores for improved exploration-exploitation balance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Base normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # UCB-style uncertainty bonus
    unc_bonus_weight = 0.5 
    unc_scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        unc_scores.append(unc_bonus_weight * sigma_sum)

    # Novelty bonus based on distance to nearest observed point
    nov_penalty_weight = 0.1 
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            # Normalize by input dimensionality (assumed to be [0, 1]^d)  
            obs_dists.append(min_dist / len(x_cand))
        nov_scores = [-nov_penalty_weight * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    # Combine all components
    final_scores = acq_scores + np.array(unc_scores) + np.array(nov_scores)
    
    return list(final_scores)