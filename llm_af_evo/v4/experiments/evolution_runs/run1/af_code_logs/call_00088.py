def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and novelty to balance exploration and exploitation."""
    
    if not context["pool"]:
        return []
        
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    campaign = context["campaign"]

    # Compute base scores
    acq_values = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Normalize uncertainties using front range  
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] 
            for name in names) 
        for cand in context["pool"]
    ]
        
    sigma_max = max(sigmas) if any(s > 0 for s in sigmars) else 1.0
    normalized_sigmas = [s/sigma_max for s in sigmas]
    
    # Compute novelty as distance to nearest observed point  
    X_obs = context["X_obs"]
    novelties = []
    for cand in context["pool"]:
        x_cand = cand["x"] 
        if len(X_obs) == 0:
            novelty = 1.0
        else:   
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            novelty = float(np.min(distances))
            
        novelties.append(novelty)

    # Normalize novelties to [0,1] scale using observed range  
    if len(context["X_obs"]) > 0:
        X_range = np.max(X_obs,axis=0) - np.min(X_obs,axis=0)
        novelty_max = float(np.linalg.norm(X_range))
        
        normalized_novelties = [
            min(novelty / novelty_max, 1.0) if novelty_max != 0 else 0 
            for novelty in novelties
        ]
    else:
        normalized_novelties = [0.] * len(context["pool"])
    
    # Progress-aware blending weights: early=more uncertainty+novelty, late=more acquisition  
    progress = campaign["progress"]
    w_acq = max(0.3 + 0.7*(1 - progress), 0.2) 
    w_uncertainty = min(0.5 * (1-progress)**0.5 , 0.4)
    w_novelty = min((1-.progress)*0.6, 0.5)

    
    scores = []
    for i in range(len(context["pool"])):
        score = (
            w_acq     * acq_values[i] +
            w_uncertainty   * normalized_sigmas[i] + 
            w_novelty       * (1 - normalized_novelties[i])
         )
        
        # Add small constant to avoid zero scores
        final_score = max(score, 1e-8)
        scores.append(final_score)

    return [float(s) for s in scores]