def modifier(context):
    """Bonus for candidates near under-covered regions of Pareto front, based on mean distance to k=3 nearestfront points."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small
    use_pf = pf.shape[0] >= 3
    ref_points = pf if use_pf else context["Y_obs"] 
    
    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all reference points 
        dists = [np.linalg.norm(mean_vec - ref_point) for ref_point in ref_points]
        dists.sort()
        
        # Take average of k=3 nearest
        avg_dist = np.mean(dists[:min(3, len(dists))])
        
        values.append(avg_dist * 0.3)
    
    return values