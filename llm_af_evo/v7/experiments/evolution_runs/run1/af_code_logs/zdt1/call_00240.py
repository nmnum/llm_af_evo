def modifier(context):
    """Suppresses candidates that are close to existing observations in feature space, encouraging exploration of unvisited regions."""
    import numpy as np
    
    # Early exit if no observations yet 
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    X_obs = context["X_obs"]
    
    values = []
    for cand in context["pool"]:
        x_cand = np.array(cand["x"]) 
        
        # Compute distances to all observed points
        diffs = X_obs - x_cand  
        dists_squared = np.sum(diffs**2, axis=1)
        
        min_dist_sq = np.min(dists_squared) 
          
        # Suppress candidates that are too close (within a dynamic threshold based on campaign progress)
        # The suppression is stronger early in the campaign and fades as it progresses
        weight = 0.5 * max(0., 1 - context["campaign"]["progress"])
        
        if min_dist_sq < 1e-6: 
            values.append(-weight)   # Strong penalty for exact duplicates  
        else:
            # Apply a soft suppression that scales inversely with distance
            suppress = weight / (min_dist_sq + 0.1)
            values.append(max(-weight, -suppress)) 
    
    return values