def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus based on distance to existing observations."""
    if not context["pool"]:
        return []
    
    scores = []
    X_obs = context["X_obs"]
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Compute squared Euclidean distance from candidate's features to nearest observed point
        dist_to_nearest = np.inf
        if len(X_obs) > 0:
            distances_sq = np.sum((X_obs - cand["x"]) ** 2, axis=1)
            dist_to_nearest = np.min(distances_sq)

        # Convert distance to a novelty score (higher is more novel): inverse of squared distance + small constant
        if dist_to_nearest == 0:
            novelty_score = float('inf')   # Perfect match — very rare but possible; treat as infinitely novel.
        else:
            novelty_score = 1. / (dist_to_nearest + 1e-8) 

        scores.append(0.9 * acq + 0.1 * novelty_score)

    return [float(score) for score in scores]