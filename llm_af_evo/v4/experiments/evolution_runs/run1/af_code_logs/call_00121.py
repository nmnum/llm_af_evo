def score_pool(context):
    """Use acquisition value as primary signal and augment with uncertainty-aware progress sensitivity."""
    names = context["objective_names"]
    
    # Normalize acq values to [0, 1] range for consistent blending
    base_acqs = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Compute normalized uncertainties 
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / (context["pareto_front_range"][name] or 1.0)  
            for name in names)
        for cand in context["pool"]
    ]
    
    sigma_max = max(sigmas) if any(s > 0 for s in sigmas) else 1.0
    normalized_sigmas = np.array([s / sigma_max for s in sigmas])
    
    # Dynamic blending based on campaign progress and stagnation 
    prog = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]

    if prog < 0.3:
        w_acq, w_sigma = 1., 0.
    elif prog > 0.7 or stagnant_batches >= 2:  
        # Emphasize uncertainty when progress stalls to encourage exploration
        w_acq, w_sigma = 0.5 + (stagnant_batches / 10.), 0.5 - (stagnant_batches / 10.)
    else:
        # Standard blending in middle stages 
        w_acq, w_sigma = 0.7, 0.3

    scores = base_acqs * w_acq + normalized_sigmas * w_sigma
    
    return list(scores)