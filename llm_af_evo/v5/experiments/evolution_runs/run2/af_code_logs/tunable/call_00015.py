def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    names = context["objective_names"]
    scores = []
    
    # Base score from normalized acquisition value  
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Add uncertainty bonus (UCB-style) 
    ucb_bonus = 0.5
    unc_scores = []
    front_range = context["pareto_front_range"]
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus * sigma_norm_sum)

    # Add novelty bonus based on distance to nearest observed point
    nov_penalty = 0.1  
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            # Normalize by the range of input space (assumed to be [0, 1]^d)  
            obs_dists.append(min_dist / len(x_cand))
        nov_scores = [-nov_penalty * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    # Combine all components
    final_scores = []
    
    total_weight = 1.0 + ucb_bonus + nov_penalty
    
    for i, _ in enumerate(context["pool"]):
        score = (acq_scores[i]
                 + unc_scores[i] 
                 + nov_scores[i])
        final_scores.append(score)
        
    return final_scores