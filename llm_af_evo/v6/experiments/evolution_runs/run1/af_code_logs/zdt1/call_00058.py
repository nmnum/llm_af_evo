def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front based on mean nearest-front-point distance."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small
    use_pf = len(pf) >= 3
    ref_points = pf if use_pf else context["Y_obs"]

    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to all reference points
        dists = [np.linalg.norm(mean_vec - ref_point) for ref_point in ref_points]
        k = min(3, len(dists))
        nearest_dists = sorted(dists)[:k] 
        mean_dist = np.mean(nearest_dists)
        
        # Scale by a fixed factor (0.2 to 0.4 range is appropriate here based on typical scales)  
        values.append(mean_dist * 0.3)

    return values