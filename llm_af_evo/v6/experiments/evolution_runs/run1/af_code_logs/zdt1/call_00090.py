def modifier(context):
    """Adaptive front proximity bonus: reward candidates near the Pareto frontier's edge to encourage boundary exploration."""
    
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    names = context['objective_names']
    
    # Compute hypervolume difference from reference point for each candidate
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]

        # Estimate the expected objective vector of this candidate 
        mean_vector = np.array([gp_posterior[name]["mean"] for name in names])
        
        if len(pareto_front) == 0:  
            # No front yet, use reference point as baseline
            hv_diff_from_ref = max(1e-8, np.prod(ref_point - mean_vector))
        else:
            # Compute distance to Pareto frontier (simplified proxy for boundary proximity)
            min_dist_to_pf = float('inf')
            
            for pf_point in pareto_front: 
                dist = np.linalg.norm(mean_vector - pf_point) / max(1e-8, np.sqrt(np.sum((ref_point - pf_point)**2)))
                
                if dist < min_dist_to_pf:
                    min_dist_to_pf = dist
                    
            # Reward candidates closer to the Pareto frontier's edge (smaller distance)
            hv_diff_from_ref = 0.5 * max(1e-8, np.prod(ref_point - mean_vector)) / (min_dist_to_pf + 1e-6)

        values.append(hv_diff_from_ref) 
        
    return values