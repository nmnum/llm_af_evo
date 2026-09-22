def score_pool(context):
    """Adjust acquisition scores based on progress-aware uncertainty scaling and regret minimization."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base acq values, uncertainties, and distances for all candidates  
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
        
        # Early: scale uncertainty more aggressively to encourage exploration
        if progress < 0.3:
            scaled_uncertainty = normalized_sigma * (1 + 2*normalized_sigma)
        else:
            # Later stages reduce the impact of raw uncertainty 
            scaled_uncertainty = normalized_sigma
            
        score = acq - 0.5 *scaled_uncertainty
        
        scores.append(score)

    return scores