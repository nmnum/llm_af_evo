def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front, based on mean nearest-front-distance."""
    import numpy as np
    
    names = context["objective_names"]
    k = 3
    bonus_factor = 0.3
    
    # Use only the Pareto front points for distance computation
    pf = context["pareto_front"]
    
    values = []
    
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        if len(pf) == 0: 
            mean_dist = 1.0
        else:
            # Compute distances to all Pareto front points and take the smallest k (or fewer)
            diffs = pf - gp_mean[None, :]
            dists_sq = np.sum(diffs**2, axis=1)
            
            sorted_indices = np.argsort(dists_sq)[:min(k, len(pf))]
            if not sorted_indices.size:
                mean_dist = 1.0
            else: 
                distances = np.sqrt(dists_sq[sorted_indices])
                mean_dist = float(np.mean(distances))
                
        # Scale by factor to control bonus magnitude, bounded
        values.append(min(bonus_factor * mean_dist, 0.5)) 
        
    return values