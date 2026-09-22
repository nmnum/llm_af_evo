def score_pool(context):
    """Incorporate uncertainty-weighted progress awareness into acquisition scores, favouring early exploration and late exploitation."""
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
    stagnation = min(context["campaign"]["stagnant_batches"], 5)
    
    # Progress-aware uncertainty weighting: explore early, exploit late
    w_explore = max(0.2, 1.0 - (3 * progress))  
    if progression > 0 and context['campaign']['n_obs'] < len(context['pool']) / 4:
        # Increase exploration bonus in initial phase with little observations 
        w_explore += min(progress*5., 0.8)
    
    for i, (acq, sigma) in enumerate(zip(acqs, sigmas)):
        normalized_sigma = sigma / sigma_max
        score = acq + w_explore * normalized_sigma
        
        scores.append(score)

    return scores