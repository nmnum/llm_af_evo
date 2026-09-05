def modifier(context):
    """Novelty-based correction that rewards candidates far from existing observations and scales with acquisition strength."""
    values = []
    x_obs = context["X_obs"]
    acq_norm = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Compute distances to nearest observed point for each candidate
    dists_to_obs = []
    for cand_x in [cand["x"] for cand in context["pool"]]:
        if len(x_obs) == 0:
            dist = float('inf')
        else:
            diffs = x_obs - cand_x  
            distances_squared = np.sum(diffs**2, axis=1)
            min_distance_sq = np.min(distances_squared)
            dist = np.sqrt(min_distance_sq)
        dists_to_obs.append(dist)

    # Normalize by the range of observed features
    if len(x_obs) > 0:
        x_min = np.min(x_obs, axis=0)
        x_max = np.max(x_obs, axis=0)
        feature_range = x_max - x_min + 1e-8  
        
    else:
        # fallback to unit range for unknown domain
        feature_range = np.ones_like(context["pool"][0]["x"])

    normalized_dists_to_obs = []
    for dist in dists_to_obs: 
        if not (np.isinf(dist) or np.isnan(dist)):
            norm_dist = min(1.0, max(0., 1. - dist / feature_range.mean()))  
        else:
            # no observations yet; assume maximum novelty
            norm_dist = 1.
        normalized_dists_to_obs.append(norm_dist)

    base_bonus_weight = 0.2
    
    for i in range(len(context["pool"])):
        bonus_term = (base_bonus_weight * 
                     normalized_dists_to_obs[i] *
                     acq_norm[i])  
        
        values.append(bonus_term)
    
    return values