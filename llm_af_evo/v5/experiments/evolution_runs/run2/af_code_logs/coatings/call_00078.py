def score_pool(context):
    """Use acquisition value as primary signal and modulate uncertainty bonus based on progress-aware decay to favor late-stage exploitation."""
    names = context["objective_names"]
    scores = []
    
    # Base score from normalized acquisition value  
    acq_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Progress-adaptive UCB-style uncertainty bonus
    campaign_progress = context["campaign"]["progress"]
    ucb_bonus_factor = 1.0 - campaign_progress  # Decays from 1 to 0 as progress increases
    
    unc_scores = []
    front_range = context["pareto_front_range"]
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        # Scale uncertainty bonus by progress-aware factor
        unc_scores.append(ucb_bonus_factor * sigma_norm_sum)

    final_scores = []
    
    for i, _ in enumerate(context["pool"]):
        score = acq_scores[i] + unc_scores[i]
        final_scores.append(score)
        
    return final_scores