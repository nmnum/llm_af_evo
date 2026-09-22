def score_pool(context):
    """Adjust acquisition scores based on progress-aware uncertainty scaling and Pareto dominance potential."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base acq values, uncertainties, and novelty for all candidates  
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        for cand in context["pool"]
    ]
    
    # Novelty: inverse distance to nearest observed point
    X_obs = context["X_obs"]
    novelties = []
    if len(X_obs) > 0:
        for cand_x in [cand["x"] for cand in context["pool"]]:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            nearest_distance = np.min(distances)
            novelty = 1. / (nearest_distance + 1e-8) 
            novelties.append(novelty)
    else:
        # No observations yet; assign equal novelty
        novelties = [1.] * len(context["pool"])
    
    scores = []
    for i, acq in enumerate(acqs):
        
        sigma_normed = sigmas[i] / (max(sigmas) or 1.)
            
        # Progress-aware scaling of uncertainty bonus  
        progress_factor = context["campaign"]["progress"]
        unc_weight = 0.3 * np.exp(-5.*progress_factor)
                
        score = acq + unc_weight*sigma_normed + 0.2*novelties[i]
        
        scores.append(score)

    return scores