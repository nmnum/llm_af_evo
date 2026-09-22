def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus to encourage exploration while prioritizing hypervolume improvement."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute distances from each candidate to the nearest observed point
    X_obs = context["X_obs"] 
    scores = []
    for cand in context["pool"]:
        x = cand["x"]
        
        # Calculate novelty as inverse of distance to closest observation  
        if len(X_obs) > 0:
            dists = np.linalg.norm(X_obs - x, axis=1)
            nearest_dist = np.min(dists)
            # Invert and normalize by feature dimensionality
            novelty_bonus = 1. / (nearest_dist + 1e-8) 
        else:  
            novelty_bonus = 1.
        
        acq_value_norm = cand["acq_value_norm"]
            
        # Blend acquisition value with a small novelty bonus
        score = acq_value_norm + 0.2 * novelty_bonus
        
        scores.append(score)
    
    return scores