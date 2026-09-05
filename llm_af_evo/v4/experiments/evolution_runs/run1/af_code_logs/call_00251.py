def score_pool(context):
    """Score candidates by combining hypervolume improvement potential with inverse novelty distance and dynamic uncertainty weighting."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute distances from all observations
    X_obs = context["X_obs"] 
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Use the precomputed acquisition value as a baseline
        acq_score = cand["acq_value_norm"]

        # Novelty: inverse of distance to nearest observed point  
        dist_to_observed = np.min(np.linalg.norm(X_obs - cand["x"], axis=1))
        novelty = 1.0 / (dist_to_observed + 1e-8)

        # Dynamic uncertainty term based on progress and front spread
        prog = context["campaign"]["progress"]
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # Weight the uncertainty component with campaign progression  
        ucb_weight = np.exp(-prog * 2.0) 
        weighted_uncertainty = ucb_weight * sigma_norm_sum
        
        scores.append(acq_score + novelty - weighted_uncertainty)

    return scores