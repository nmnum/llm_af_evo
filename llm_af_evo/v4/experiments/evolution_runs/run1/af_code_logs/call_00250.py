def score_pool(context):
    """Exploit acquisition value early, switch to uncertainty-aware exploration as campaign progresses."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Early exploitation: use acq_value_norm directly with a bonus for high uncertainty
    # Late phase: emphasize candidates that are both promising and uncertain (UCB-style)
    progress = context["campaign"]["progress"] 
    scores = []
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        sigma_sum = sum(cand["gp_posterior"][name]["std"] for name in names)

        # Progress-aware blending: early on, prefer high acquisition; later, add more uncertainty bonus
        weight_acquisition = max(0.7 - progress * 0.5, 0.2)
        
        score = acq * weight_acquisition + (1 - weight_acquisition) * sigma_sum
        
        scores.append(score)

    return scores