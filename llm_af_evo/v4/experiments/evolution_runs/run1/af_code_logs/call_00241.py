def score_pool(context):
    """Boost acquisition scores for candidates with low uncertainty and high novelty, emphasizing diverse exploration early then focused exploitation later."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base acq values and uncertainties for all candidates  
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        for cand in context["pool"]
    ]
    
    # Normalize uncertainties to 0-1 scale
    sigma_max = max(sigmas) if any(s > 0 for s in sigmas) else 1.0
    
    scores = []
    progress = context["campaign"]["progress"] 
    
    for i, (acq, sigma) in enumerate(zip(acqs, sigmas)):
        normalized_sigma = sigma / sigma_max
        
        # Early exploration: prefer low uncertainty candidates
        if progress < 0.5:
            score_boost = max(1e-6, 1 - normalized_sigma)
        else:
            # Later exploitation: reward high acquisition value with moderate penalty for noise  
            score_boost = acq * (1 + 2*(normalized_sigma))
            
        scores.append(acq * score_boost) 
    
    return scores