def score_pool(context):
    """Blend acquisition value with uncertainty and novelty to improve diversity and exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base scores using acq_value_norm + UCB-style bonus
    raw_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        acq = cand["acq_value_norm"] 
        ucb_bonus = 0.1 * sum(gp[name]["std"] / front_range[name] for name in names)
        raw_scores.append(acq + ucb_bonus)

    # Apply novelty adjustment to reduce score of nearby candidates
    scores = []
    x_observed = context["X_obs"]
    
    if len(x_observed) == 0:
        return raw_scores

    for i, cand in enumerate(context["pool"]):
        min_dist_x = float('inf')
        
        # Find minimum Euclidean distance to any observed point
        for obs_point in x_observed:  
            dist = np.linalg.norm(cand['x'] - obs_point)
            if dist < min_dist_x:
                min_dist_x = dist

        # Scale down score based on inverse of distance (novelty penalty) 
        novelty_factor = 1.0 / (1.0 + min_dist_x * 5.0)  
        scores.append(raw_scores[i] * novelty_factor)
        
    return scores