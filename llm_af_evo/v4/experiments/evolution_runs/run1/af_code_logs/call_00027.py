def score_pool(context):
    """Blend acquisition value with uncertainty-aware novelty and progress-driven exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base acq values for all candidates  
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Normalize acquisition scores to 0-1 scale
    if max(acqs) > min(acqs):
        normalized_acqs = [(a - min(acqs)) / (max(acqs) - min(acqs)) for a in acqs]
    else:
        normalized_acqs = [0.5] * len(acqs)
    
    # Compute uncertainties and normalize
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names) 
        for cand in context["pool"]
    ]
    
    sigma_max = max(sigmas) if any(s > 0 for s in sigmas) else 1.0
    normalized_sigmas = [s / sigma_max for s in sigmas]
        
    # Progress-aware blending: early exploration, late exploitation  
    progress = context["campaign"]["progress"]
    
    # Use a sigmoid to smoothly transition from uncertainty-heavy (early) 
    # to acquisition-value heavy (late)
    blend_factor = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))
        
    scores = []
    for i, (_, normalized_acq, sigma_norm) in enumerate(zip(acqs, normalized_acqs, normalized_sigmas)):
        # Blend acquisition and uncertainty based on progress
        score = blend_factor * normalized_acq + (1.0 - blend_factor) * sigma_norm
        
        scores.append(score)
    
    return scores