def modifier(context):
    """Add a repulsion bonus based on distance to already-observed points, scaled by acquisition strength."""
    import numpy as np
    
    pool = context["pool"]
    X_obs = context["X_obs"] 
    names = context['objective_names']
    
    values = []
    
    for cand in pool:
        # Compute squared Euclidean distances from candidate x to all observed points
        dists_sq = np.sum((X_obs - cand["x"]) ** 2, axis=1)
        
        # Use the minimum distance (avoid division by zero) 
        min_dist_sq = np.min(dists_sq)
        
        if min_dist_sq < 1e-8:  
            repulsion_bonus = 0.0
        else:
            # Invert and scale to get a bonus for being far from observed points,
            # but only apply it moderately (scale factor) 
            repulsion_bonus = 0.2 / np.sqrt(min_dist_sq)
            
        # Scale the repulsion by how acquisition-strong this candidate already is  
        acq_value_norm = cand["acq_value_norm"]
        
        values.append(repulsion_bonus * acq_value_norm)

    return values