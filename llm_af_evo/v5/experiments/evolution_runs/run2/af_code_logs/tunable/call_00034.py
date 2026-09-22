def score_pool(context):
    """Blend acquisition value with uncertainty-aware and novelty-informed scoring for dynamic exploration-exploitation."""
    names = context["objective_names"]
    scores = []
    
    # Base score from normalized acquisition value  
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Add UCB-style bonus based on uncertainty
    ucb_bonus_weight = 0.5
    
    front_range = context["pareto_front_range"]
    unc_scores = []
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus_weight * sigma_norm_sum)

    # Add novelty bonus based on distance to nearest observed point
    nov_penalty_weight = 0.1  
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            # Normalize by the range of input space (assumed to be [0, 1]^d)  
            obs_dists.append(min_dist / len(x_cand))
        nov_scores = [-nov_penalty_weight * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    # Combine all components
    final_scores = []
    
    for i, _ in enumerate(context["pool"]):
        score = (acq_scores[i]
                 + unc_scores[i] 
                 + nov_scores[i])
        final_scores.append(score)
        
    return final_scores