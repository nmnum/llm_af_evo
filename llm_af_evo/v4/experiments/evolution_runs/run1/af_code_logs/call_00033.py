def score_pool(context):
    """Integrate acquisition value with uncertainty-aware progress sensitivity and novelty to balance exploration and exploitation."""
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute base scores from botorch's acquisition value
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Normalize uncertainty across all candidates 
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        for cand in context["pool"]
    ]
    sigma_max = max(sigmas) if any(s > 0 for s in sigmars) else 1.0
    normalized_sigmas = [s / sigma_max for s in sigmas]
    
    # Compute novelty scores based on distance to existing observations  
    X_obs = context["X_obs"]
    novelties = []
    for cand_x in (cand["x"] for cand in context["pool"]):
        if len(X_obs) == 0:
           novelty = 1.0
        else: 
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty = float(np.exp(-np.min(distances)))
        novelties.append(novelty)

    # Blend acquisition value with normalized uncertainty and novelty  
    progress_factor = context["campaign"]["progress"]
    
    scores = []
    for acq_val, sigma_norm, novelty in zip(acq_values, normalized_sigmas, novelties):
        
        # Progress-aware weighting between exploitation (acquisition) vs exploration (uncertainty + novelty)
        w_exploit = 0.5 * np.exp(-2*progress_factor) 
        w_explor = 1 - w_exploit
        
        score = w_exploit * acq_val + \
                w_explor * sigma_norm * novelty

        scores.append(score)

    return scores