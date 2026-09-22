def modifier(context):
    """Add a diversity bonus based on repulsion from top candidates in current batch, encouraging exploration away from already selected high-acquisition points."""
    import numpy as np
    
    pool = context["pool"]
    
    # Get acquisition values and candidate features for the current pool 
    acq_values = [cand['acq_value_norm'] for cand in pool]
    X_pool = np.array([cand['x'] for cand in pool])
    
    if len(pool) < 2:
        return [0.0] * len(pool)
        
    # Select top candidates by acquisition value (e.g., top 30% or so, but at least 1)
    n_top = max(1, int(len(pool) * 0.3))
    
    if n_top >= len(pool):
        return [0.0] * len(pool)

    sorted_indices = np.argsort(acq_values)[::-1][:n_top]
    X_top = X_pool[sorted_indices]

    # Compute pairwise distances between top candidates
    dist_matrix = np.sqrt(np.sum((X_top[:, None, :] - X_top[None, :, :]) ** 2, axis=2))
    
    # Avoid division by zero; set diagonal to large value so self-distance doesn't affect repulsion  
    np.fill_diagonal(dist_matrix, np.inf)
        
    # Compute inverse distance-based suppression (higher penalty for closer points) 
    inv_distances = 1.0 / dist_matrix
    suppressions = np.sum(inv_distances, axis=1)

    values = []
    
    for i in range(len(pool)):
        cand_x = X_pool[i]
                
        distances_to_top = [np.linalg.norm(cand_x - top_cand) 
                            for top_cand in X_top] 
        
        # If candidate is far from all top candidates, minimal suppression
        if np.min(distances_to_top) > 0.1:  
            values.append(0.0)
            
        else:
            # Apply inverse-distance repulsion bonus (more diverse points get higher bonuses)
            total_repulse = sum([inv_distances[j][i] for j in range(len(X_top))])
            penalty_factor = min(total_repulse / 5, 1) 
                
            values.append(-penalty_factor * 0.2)

    return values