def score_pool(context):
    """Dynamically adjust exploration-exploitation balance using uncertainty scaling and progress-aware novelty rewards."""
    names = context["objective_names"]
    scores = []
    
    # Base acquisition value from Botorch's qLogNEHVI 
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Progress-aware UCB bonus with decaying exploration
    campaign_progress = context["campaign"]["progress"]
    ucb_bonus_weight = 1.0 - campaign_progress  # Less exploration as we progress
    
    unc_scores = []
    front_range = context["pareto_front_range"] 
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        # Scale bonus by progress and apply UCB-style uncertainty
        unc_scores.append(ucb_bonus_weight * 2.0 * sigma_norm_sum)

    # Novelty reward based on distance to nearest observed point, modulated by campaign state  
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        obs_dists = []
        
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist = np.min(dists)

            # Normalize by input space dimensionality (assumed to be [0, 1]^d)  
            normalized_distance = min_dist / len(x_cand)
            
            obs_dists.append(normalized_distance)
        
        nov_scores = []
        for dist in obs_dists:
            # Reward novelty more heavily early on and less later
            reward_factor = (1.0 - campaign_progress)**2 
            nov_score = -(reward_factor * 0.5) * dist  
            nov_scores.append(nov_score)

    else:   # No observations yet, no novelty penalty applied
        nov_scores = [0.] * len(context["pool"])

    final_scores = []
    
    for i in range(len(context["pool"])):
        
        score = (acq_scores[i] 
                 + unc_scores[i]
                 + nov_scores[i])
                
        final_scores.append(score)
      
    return final_scores