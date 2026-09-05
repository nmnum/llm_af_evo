def score_pool(context):
    """Exploit acquisition value early, switch to uncertainty-driven exploration as campaign progresses."""
    progress = context["campaign"]["progress"]
    
    # Early phase: prefer high acq_value_norm with slight bonus for uncertainty
    # Late phase: favor candidates with higher uncertainty (UCB-style)
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"] 
        sigma = sum(cand["gp_posterior"][name]["std"] for name in context["objective_names"])
        
        if progress < 0.5:
            # Early: balance acquisition and uncertainty
            score = acq + (1 - progress) * sigma / 2.
        else:
            # Late: prefer uncertain candidates 
            score = acq + (progress - 0.5) * sigma
            
        scores.append(score)
    
    return scores