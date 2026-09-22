def modifier(context):
    """Bonus for candidates in under-covered regions of Pareto front based on mean nearest-front-distance with adaptive bonus scaling."""
    names = context["objective_names"]
    k = 3
    
    pf = context["pareto_front"]
    
    # Use all observations and pareto points to compute distances
    if len(pf) < k:
        use_points = np.vstack([context["Y_obs"], pf])
    else:
        use_points = pf

    values = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])

        if len(use_points) == 0: 
            mean_dist = float('inf')
        else:
            diffs = use_points - gp_mean[None, :]
            dists_sq = np.sum(diffs**2, axis=1)
            sorted_indices = np.argsort(dists_sq)[:min(k, len(use_points))]
            
            if not sorted_indices.size:
                mean_dist = 1.0
            else: 
                distances = np.sqrt(dists_sq[sorted_indices])
                mean_dist = float(np.mean(distances))
        
        # Adaptive bonus scaling with campaign progress to encourage exploration early on  
        adaptive_factor = (1 - context["campaign"]["progress"]) * 0.3
        
        values.append(min(adaptive_factor * mean_dist, 0.5))

    return values