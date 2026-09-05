def score_pool(context):
    """Exploit acquisition value early with dynamic uncertainty weighting, then shift to novelty-driven selection as campaign progresses."""
    names = context["objective_names"]
    
    # Base scores from botorch's qLogNEHVI values  
    base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Compute normalized uncertainties per candidate
    front_range = context["pareto_front_range"]
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] 
            for name in names) 
        for cand in context["pool"]
    ]
    
    # Normalize sigma values to [0, 1]
    max_sigma = max(sigmas) if any(s > 0 for s in sigmas) else 1.0
    normalized_sigmas = [s / max_sigma for s in sigmas]

    campaign_progress = context["campaign"]["progress"]
    
    # Dynamic weighting: early exploitation + late exploration  
    w_acq = max(0.7, 1.0 - campaign_progress * 0.5)
    w_uncertainty = min(0.3, campaign_progress * 0.6) 
    
    scores = [
        w_acq * base_score +
        (w_uncertainty + normalized_sigma / len(context["pool"])) 
        for base_score, normalized_sigma in zip(base_scores, normalized_sigmas)
    ]
    
    return scores