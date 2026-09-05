def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus to encourage exploration while prioritizing hypervolume improvement."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    
    # Compute squared distances from each candidate x to the nearest observed point
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]  # (d,)
        
        if len(X_obs) == 0:
            dist_to_nearest = float('inf')
        else:
            diff = X_obs - x_cand[None, :]   # (n_obs, d)
            distances_sq = np.sum(diff ** 2, axis=1)  
            dist_to_nearest = np.sqrt(np.min(distances_sq))
        
        acq_norm = cand["acq_value_norm"]
        
        novelty_bonus = min(0.5 * dist_to_nearest / max(front_range.values()), 1.)
        
        score = acq_norm + 0.2 * novelty_bonus
        scores.append(score)
    
    return scores