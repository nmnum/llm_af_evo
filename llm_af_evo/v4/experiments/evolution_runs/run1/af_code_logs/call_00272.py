def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus based on distance to nearest observed point."""
    scores = []
    X_obs = context["X_obs"]
    
    for cand in context["pool"]:
        acq_norm = cand["acq_value_norm"] 
        x_cand = cand["x"]

        # Compute min Euclidean distance from candidate to any previously observed point
        if len(X_obs) > 0:
            distances_sq = np.sum((X_obs - x_cand)**2, axis=1)
            novelty_bonus = np.sqrt(np.min(distances_sq))
        else:
            # If no observations yet, assign high bonus for exploration 
            novelty_bonus = 1.0

        # Blend acquisition value with a small novelty term (weighting controlled by campaign progress)  
        blend_weight = max(0.2, 0.8 - context["campaign"]["progress"] * 0.6)
        
        score = acq_norm + blend_weight * novelty_bonus
        scores.append(score)

    return scores