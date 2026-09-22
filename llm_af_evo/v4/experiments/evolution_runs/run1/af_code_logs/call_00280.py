def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a novelty bonus based on distance to nearest observed point."""
    scores = []
    X_obs = context["X_obs"]
    pool = context["pool"]
    
    for cand in pool:
        acq_val = cand['acq_value_norm']
        
        # Compute squared Euclidean distance to the nearest observation
        x_cand = cand["x"]
        distances_sq = np.sum((X_obs - x_cand) ** 2, axis=1)
        min_distance_sq = np.min(distances_sq)
        
        # Novelty bonus: higher when candidate is farther from any observed point
        novelty_bonus = 0.5 * (min_distance_sq + 1e-8)**(-0.5)
        
        score = acq_val + 0.2 * novelty_bonus
        scores.append(score)

    return scores