def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front based on mean distance to k=3 nearestfront points."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use pareto_front if it has sufficient points, otherwise fall back to Y_obs
    pf_used = context["pareto_front"] 
    n_pf = len(pf_used)
    
    if n_pf < 3:
        # Fall back to all observations for early campaign stages  
        obs_points = np.array(context["Y_obs"])
        reference_set = obs_points
    else:   
        reference_set = pf_used
    
    k = min(3, len(reference_set))
        
    values = []
    
    for cand in context["pool"]:
        # Get candidate's mean prediction vector (already flipped if needed)
        pred_mean_vec = np.array([cand["gp_posterior"][name]["mean"] 
                                 for name in names])
            
        distances = []  
        for point in reference_set:
            dist = np.linalg.norm(pred_mean_vec - point)  # Euclidean distance
            distances.append(dist)

        sorted_dists = sorted(distances)
        
        mean_dist_to_k_nearest = sum(sorted_dists[:k]) / k
        
        # Scale by a bounded factor (0.3 is chosen as an example, could be tuned per domain)
        bonus_factor = 0.3
        values.append(mean_dist_to_k_nearest * bonus_factor)

    return values