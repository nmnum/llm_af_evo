def modifier(context):
    """Penalize candidates that are close to higher-ranked already-selected candidates, discouraging batch duplication."""
    import numpy as np
    
    # Get top-k ranked candidates from previous batches (if any)
    if not hasattr(modifier, 'selected_candidates'):
        selected_candidates = []
    else:
        selected_candidates = modifier.selected_candidates
        
    names = context["objective_names"]
    
    penalty_factor = 0.5
    max_penalty = 0.3
    
    values = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all previously selected candidates (if any)
        if len(selected_candidates) == 0:
            penalty = 0.0
        else:
            diffs = np.vstack(selected_candidates) - gp_mean[None, :]
            dists_sq = np.sum(diffs**2, axis=1)
            min_dist_squared = np.min(dists_sq)
            
            # Convert to actual distance and apply a soft threshold for penalty 
            if min_dist_squared < 0.05:   # Threshold in normalized space
                penalty = penalty_factor * (1 - min_dist_squared / 0.05)  
            else:
                penalty = 0.0
                
        values.append(-min(penalty, max_penalty))
        
    return values