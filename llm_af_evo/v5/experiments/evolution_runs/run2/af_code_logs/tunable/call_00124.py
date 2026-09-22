def score_pool(context):
    """Incorporate progress-aware uncertainty scaling and adaptive acquisition blending to dynamically balance exploration and exploitation."""
    names = context["objective_names"]
    scores = []
    
    # Base normalized acquisition value  
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Dynamic UCB bonus that decreases as campaign progresses
    progress = context["campaign"]["progress"]
    ucb_bonus = 1.0 * (1 - progress) 
    
    unc_scores = []
    front_range = context["pareto_front_range"]
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus * sigma_norm_sum)

    # Adaptive blend: higher acquisition weight early, more uncertainty-weighted later
    acq_weight = 0.7 + 0.3 * (1 - progress)  
    unc_weight = 1.0 - acq_weight
    
    final_scores = []
    
    for i, _ in enumerate(context["pool"]):
        score = acq_weight * acq_scores[i] + unc_weight * unc_scores[i]
        final_scores.append(score)
        
    return final_scores