def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus based on distance to nearest observed point."""
    scores = []
    X_obs = context["X_obs"]
    pool = context["pool"]
    
    for cand in pool:
        acq_value = cand["acq_value_norm"] 
        x_cand = cand["x"]
        
        # Compute Euclidean distance to the closest previously observed point
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            novelty_bonus = np.min(distances)
        else:
            novelty_bonus = 1.0  
            
        score = acq_value + 0.2 * novelty_bonus 
        scores.append(score)

    return scores