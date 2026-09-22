def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus based on distance to nearest observed point."""
    scores = []
    X_obs = context["X_obs"]
    pool = context["pool"]
    
    for cand in pool:
        acq_norm = cand["acq_value_norm"] 
        x_cand = cand["x"]
        
        # Compute Euclidean distances from candidate to all observations
        dists = np.linalg.norm(X_obs - x_cand, axis=1)
        min_dist = np.min(dists) if len(dists) > 0 else 0.0
        
        novelty_bonus = 2 * (1 - min_dist / (np.sqrt(len(x_cand)) + 1e-8))
        
        score = acq_norm + 0.3 * novelty_bonus
        scores.append(score)
    
    return scores