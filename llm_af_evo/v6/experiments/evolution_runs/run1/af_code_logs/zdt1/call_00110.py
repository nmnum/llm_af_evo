def modifier(context):
    """Bonus for candidates near under-covered regions of Pareto front based on mean distance to k=3 nearestfront points."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Use pareto_front if it has sufficient points, otherwise fall back to Y_obs
    pf_points = context["pareto_front"] 
    if len(pf_points) < 3:
        ref_set = context["Y_obs"]
    else:
        ref_set = pf_points
    
    values = []
    k = min(3, len(ref_set))
    
    for cand in context["pool"]:
        mean_vector = np.array([cand["gp_posterior"][name]["mean"] 
                               for name in names])
        
        # Compute distances to all reference points
        dists = [np.linalg.norm(mean_vector - ref_point) 
                for ref_point in ref_set]
                
        # Get k nearest (sorted)
        sorted_dists = np.sort(dists)[:k] 
        
        mean_dist = np.mean(sorted_dists)
        
        # Scale by a bounded factor to keep correction terms moderate
        bonus = 0.3 * mean_dist 
        values.append(bonus) 

    return values