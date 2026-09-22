def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus to improve exploration."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute squared distances from each candidate to the nearest observed point
    X_obs = context['X_obs']
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        # Calculate novelty as inverse of distance to closest observation (avoiding division by zero)
        if len(X_obs) > 0:
            distances_sq = np.sum((X_obs - x_cand)**2, axis=1)
            min_distance_sq = np.min(distances_sq)
            
            # Add small epsilon for numerical stability
            novelty_bonus = 1. / (min_distance_sq + 1e-8)
        else:
            # If no observations yet, assign high initial bonus to encourage exploration 
            novelty_bonus = 10.
        
        acq_value_norm = cand["acq_value_norm"]
            
        # Combine acquisition value and novelty with a small weight on the latter
        score = acq_value_norm + 0.2 * novelty_bonus
        
        scores.append(score)
    
    return scores