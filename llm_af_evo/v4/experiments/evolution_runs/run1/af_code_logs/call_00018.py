def score_pool(context):
    """Incorporate uncertainty-aware novelty into acquisition scores, favouring diverse high-expected-improvement points."""
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
    for i, (acq, sigma) in enumerate(zip(acqs, sigmas)):
        normalized_sigma = sigma / sigma_max
        
        # Blend acquisition value with uncertainty-aware novelty
        score = acq + 0.2 * normalized_sigma

        scores.append(score)
    
    return scores