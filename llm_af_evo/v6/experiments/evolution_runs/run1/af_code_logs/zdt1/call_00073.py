def modifier(context):
    """Add an adaptive uncertainty bonus scaled by the inverse of recent improvement rate and candidate's distance to the current Pareto front."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    names = context['objective_names']
    
    # Adaptive weight based on stagnation
    stagnant_batches = context["campaign"]["stagnant_batches"]  
    base_weight = 0.4 * (1.0 - min(stagnant_batches / 5.0, 1.0))
    
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Compute uncertainty bonus
        sigma_norm = sum(gp_posterior[name]["std"] for name in names)
        
        # Estimate distance to Pareto front (minimum Euclidean distance) 
        min_dist_sq = float('inf')
        x_pred = np.array([gp_posterior[name]["mean"] for name in names])
    
        for pf_point in pareto_front:
            dist_sq = sum((x - y)**2 for x, y in zip(x_pred, pf_point))
            if dist_sq < min_dist_sq: 
                min_dist_sq = dist_sq
                
        # Distance bonus (larger when farther from front)
        distance_bonus = np.sqrt(min_dist_sq) / max(1e-6, len(names))  # Normalized
        
        values.append(base_weight * sigma_norm + 0.25 * distance_bonus)

    return values