def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted progress awareness to favor candidates that are both highly promising and sufficiently novel."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # Progress-aware uncertainty: reduce influence of noise as campaign progresses
        progress_factor = 1.0 - context["campaign"]["progress"]
        adjusted_sigma = sigma_sum * (0.5 + 0.5 * progress_factor)  
        
        # Blend acquisition with a scaled, progressive penalty for high uncertainty 
        score = acq * (1.0 + 2.0 *adjusted_sigma)
        scores.append(score)

    return scores