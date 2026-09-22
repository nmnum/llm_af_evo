def score_pool(context):
    """Integrate acquisition value with a dynamic tradeoff between uncertainty and novelty based on campaign stagnation."""
    names = context["objective_names"]
    scores = []
    
    # Base score from normalized acquisition value  
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"]

    unc_scores = [] 
    nov_scores = []
    
    # Compute uncertainty scores (UCB-style)
    ucb_weight = 0.5 if stagnant_batches > 2 else 1.0
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_weight * sigma_norm_sum)

    # Compute novelty scores (inverse distance to nearest observed point)  
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)
            # Normalize by the range of input space (assumed to be [0, 1]^d)  
            obs_dists.append(min_dist / len(x_cand))
        
        nov_weight = 0.3 if stagnant_batches > 2 else 0.1
        nov_scores = [-nov_weight * d for d in obs_dists]
    else:
        nov_scores = [0.0] * len(context["pool"])
    
    # Combine scores with dynamic blending based on stagnation
    final_scores = []
    for i, _ in enumerate(context["pool"]):
        if stagnant_batches > 2: 
            score = acq_scores[i] + unc_scores[i]
        else:
            score = acq_scores[i] + nov_scores[i]

        final_scores.append(score)
        
    return final_scores