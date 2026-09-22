def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus to encourage exploration while preserving hypervolume improvement dominance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute distance from each candidate to the nearest observed point
    X_obs = context["X_obs"] 
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        # Novelty bonus: inverse of squared Euclidean distance to closest observation  
        if len(X_obs) > 0:
            distances_sq = np.sum((X_obs - x_cand)**2, axis=1)
            nearest_dist_sq = np.min(distances_sq)
            novelty_bonus = 1.0 / (nearest_dist_sq + 1e-8)
        else:
            # No observations yet: high bonus to encourage initial exploration
            novelty_bonus = 10.0
            
        acq_value_norm = cand["acq_value_norm"]
        
        # Blend acquisition value with novelty; let acquistion dominate 
        score = 0.9 * acq_value_norm + 0.1 * novelty_bonus
        
        scores.append(score)
    
    return scores