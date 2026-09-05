def modifier(context):
    """Bonus for candidates near under-covered regions of Pareto front, based on mean distance to k=3 nearestfront points."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use pareto_front if it has sufficient points, otherwise fall back to Y_obs
    pf_used = context['pareto_front']
    if len(pf_used) < 3:
        obs_points = context['Y_obs'] 
        all_points = np.vstack([pf_used, obs_points])  
    else:   
        all_points = pf_used
    
    # Prepare for distance calculations (normalize by front range)
    ranges = [context["pareto_front_range"][name] for name in names]
    
    values = []
    k = 3
    for cand in context['pool']:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Normalize the candidate and points using front ranges  
        norm_cand = mean_vec / ranges
        
        all_norm_points = all_points[:, :len(names)]  / ranges
          
        dists = [np.linalg.norm(norm_cand - p) for p in all_norm_points]
            
        nearest_dists = sorted(dists)[:k] 
        
        bonus = np.mean(nearest_dists)
        
        values.append(0.3 * bonus)

    return values