def modifier(context):
    """Penalty for candidates that are close to already-observed points, with adaptive scaling based on observation density and campaign progress."""
    
    if len(context["X_obs"]) == 0:
        return [0.0] * len(context["pool"])
        
    names = context["objective_names"]
    # Normalize features
    x_min = np.min(context["X_obs"], axis=0)
    x_max = np.max(context["X_obs"], axis=0) 
    ranges_x = x_max - x_min
    normalized_X_obs = (context["X_obs"] - x_min) / ranges_x
    
    values = []
    
    # Determine penalty scaling based on campaign progress and observation density  
    n_obs = len(context["X_obs"])
    current_progress = context["campaign"]["progress"]
    max_penalty_factor = 0.5 * (1.0 - current_progress)
    
    for cand in context["pool"]:
        x_cand = np.array(cand["x"]) 
        normalized_x = (x_cand - x_min) / ranges_x
        
        # Compute distances to all observed points
        diffs = normalized_X_obs - normalized_x[None, :]
        dists_sq = np.sum(diffs**2, axis=1)
        
        min_dist_squared = np.min(dists_sq)
        if min_dist_squared == 0.0:
            penalty = max_penalty_factor * 3.0  
        else: 
            # Inverse scaling with squared distance
            inv_distance = 1.0 / (min_dist_squared + 1e-8)   
            penalty = max Penalty factor * inv_distance
            
        values.append(-penalty)
        
    return values