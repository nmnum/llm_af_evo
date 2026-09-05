def score_pool(context):
    """Blend acquisition value with a progress-aware uncertainty term and novelty bonus to balance exploration and exploitation."""
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point"]
    campaign = context["campaign"]

    # Progress-based weighting for UCB-style tradeoff
    progress = campaign["progress"]
    exploit_weight = 0.3 + 0.7 * (1 - progress)  # Decrease exploitation as we approach end
    
    scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        
        acq_val_norm = cand["acq_value_norm"]
        
        # UCB-style uncertainty term, scaled by campaign state
        sigma_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                        for name in names)
                        
        ucb_term = exploit_weight * (1 - progress) * sigma_sum
        
        # Novelty bonus: distance to nearest previously observed point  
        cand_x = np.array(cand["x"])
        
        if len(context["X_obs"]) > 0:
            distances = [np.linalg.norm(cand_x - obs_x, ord=2) 
                         for obs_x in context["X_obs"]]
            min_distance = float(np.min(distances))
            
            # Normalize by feature space extent (assuming unit hypercube)
            novelty_bonus = np.exp(-min_distance * 10.0)  
        else:
            novelty_bonus = 0.0
            
        final_score = acq_val_norm + ucb_term + 0.5 * novelty_bonus
        scores.append(final_score)

    return scores