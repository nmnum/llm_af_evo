def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front, based on mean nearest-front-distance."""
    import numpy as np
    
    names = context["objective_names"]
    k = 3
    bonus_factor = 0.2719
    
    # Prepare points for distance computation: use pareto_front if it has enough points,
    # otherwise fall back to Y_obs (excluding those already in the front)
    pf = context["pareto_front"]
    
    if len(pf) < k:
        # Not enough points on Pareto front, so include observations
        obs_points = np.vstack([context["Y_obs"], pf])  # Add current PF to observed points for diversity
        use_points = obs_points[np.unique(np.concatenate((np.arange(len(obs_points)), 
                                                          [i for i in range(len(pf)) if any(np.allclose(pf[i], p) for p in obs_points)]), 
                                                         axis=None, return_index=True)[1])]
    else:
        # Use only the Pareto front
        use_points = pf
    
    values = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(use_points) == 0: 
            dists = [np.inf] * k
        else:
            # Compute distances to all points and take the smallest k (or fewer)
            diffs = use_points - gp_mean[None, :]
            dists_sq = np.sum(diffs**2, axis=1)
            
            sorted_indices = np.argsort(dists_sq)[:min(k, len(use_points))]
            if not sorted_indices.size:
                # Shouldn't happen but just in case
                dists = [np.inf] * k  
            else: 
                dists = list(np.sqrt(dists_sq[sorted_indices]))
                
        mean_dist = np.mean(dists) if dists and len(dists)>0 else 1.0
        
        # Scale by factor to control bonus magnitude, bounded
        values.append(min(bonus_factor * mean_dist, 0.5078)) 
        
    return values