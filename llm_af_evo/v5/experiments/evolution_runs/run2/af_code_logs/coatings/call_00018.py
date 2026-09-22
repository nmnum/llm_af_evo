def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Uncertainty bonus (UCB-style) 
    ucb_bonus_weight = 0.3
    front_range = context["pareto_front_range"]
    unc_scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus_weight * sigma_norm_sum)

    # Novelty penalty based on distance to nearest observed point
    nov_penalty_weight = 0.2  
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            # Normalize by the number of features
            obs_dists.append(min_dist / len(x_cand))
        nov_scores = [-nov_penalty_weight * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    final_scores = acq_scores + np.array(unc_scores) + np.array(nov_scores)
    
    return list(final_scores)