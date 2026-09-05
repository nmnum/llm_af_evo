def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus based on distance to nearest observed point."""
    scores = []
    x_obs = context["X_obs"]
    pool = context["pool"]
    
    for cand in pool:
        acq_norm = cand["acq_value_norm"] 
        dist_to_nearest = np.min(np.linalg.norm(x_obs - cand["x"], axis=1))
        
        novelty_bonus = 0.5 * (1.0 / (dist_to_nearest + 1e-8))  
        scores.append(acq_norm + 0.2 * novelty_bonus)
    
    return scores