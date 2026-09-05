def score_pool(context):
    """Score candidates by acquisition value adjusted for proximity-based diversity and uncertainty-aware exploitation."""
    names = context["objective_names"]
    X_obs = context["X_obs"] 
    pool_size = len(context["pool"])
    
    # Normalize all features to [0,1] range if not already
    x_min = np.min(X_obs, axis=0)
    x_max = np.max(X_obs, axis=0)  
    eps = 1e-8
    
    def normalize(x):
        return (x - x_min) / (x_max - x_min + eps)

    # Normalize pool features
    normalized_pool_x = [normalize(cand["x"]) for cand in context["pool"]]
    
    if len(X_obs.shape) == 1:
        X_obs_normed = np.expand_dims(normalize(X_obs), axis=0)
    else: 
        X_obs_normed = normalize(X_obs)

    # Compute distances from each candidate to all observed points
    scores = []
    for i, cand in enumerate(context["pool"]):
        acq_val = cand['acq_value_norm']
        
        gp_posterior = cand["gp_posterior"]
        mu_sum = sum(gp_posterior[name]["mean"] for name in names)
        sigma_sum = np.sum([gp_posterior[name]["std"] for name in names])
                
        # Compute distance to nearest observed point
        dist_to_observed = float('inf')
        
        if len(X_obs_normed) > 0:
            cand_x_normalized = normalized_pool_x[i]
            
            distances_squared = np.sum((X_obs_normed - cand_x_normalized)**2, axis=1)
            min_dist_sq = np.min(distances_squared)
                
            dist_to_observed = np.sqrt(min_dist_sq)

        # Inverse of distance (higher is better) as novelty bonus
        if dist_to_observed < 1e-8:
            novel_bonus = float('inf')
        else:  
            novel_bonus = 1.0 / max(dist_to_observed, eps)
            
        final_score = acq_val + sigma_sum * 0.5 - (novel_bonus ** 2) * 0.3
        
        scores.append(final_score)

    return scores