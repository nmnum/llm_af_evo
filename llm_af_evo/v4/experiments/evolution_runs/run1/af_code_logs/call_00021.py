def score_pool(context):
    """Blend acquisition value with uncertainty-aware novelty, but suppress scores for candidates that are too close to already observed points."""
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
    
    X_obs = context["X_obs"] 
    dists_to_observed = np.array([
        np.linalg.norm(X_obs - cand["x"], axis=1).min()
        for cand in context["pool"]
    ])
    
    # Normalize distances to [0,1] scale
    d_min, d_max = dists_to_observed.min(), dists_to_observed.max() 
    if d_max > d_min:
        nov_norm = (dists_to_observed - d_min) / (d_max - d_min)
    else:
        # All candidates are at same point; use a default value
        nov_norm = np.full(len(dists_to_observed), 0.5)

    scores = []
    for i, (acq, sigma, dist) in enumerate(zip(acqs, sigmas, dists_to_observed)):
        normalized_sigma = sigma / sigma_max
        
        # Blend acquisition with uncertainty-aware novelty
        score = acq + 0.2 * normalized_sigma

        # Suppress candidates that are too close to observed points (novelty penalty)
        if dist < np.percentile(dists_to_observed, 15):  
            suppression_factor = max(0., 1 - 3*(dist / np.percentile(dists_to_observed, 20)))
            score *= suppression_factor

        scores.append(score)

    return scores