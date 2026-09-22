def modifier(context):
    """Add an adaptive uncertainty bonus based on how much the candidate's prediction varies from the current Pareto front’s objective ranges."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    names = context['objective_names']
    values = []
    
    # Compute reference point normalized by range for each objective
    front_range = [context["pareto_front_range"][name] for name in names]
        
    for cand in pool:
        gp_posterior = cand["gp_posterior"]

        # Calculate how far the candidate's mean is from current Pareto frontier (normalized)
        means = np.array([gp_posterior[name]["mean"] for name in names])
        distances_to_front = []
        if len(pareto_front) > 0:
            # For each objective, compute distance to nearest non-dominated point
            min_distances = []  
            for i, mean_val in enumerate(means):
                front_vals = pareto_front[:,i]
                dists = np.abs(mean_val - front_vals)
                if len(dists[dists >= 0]) > 0:
                    # Distance is positive difference from the closest non-dominated point
                    min_dist = np.min(np.where(dists >= 0, dists, np.inf))
                    min_distances.append(min_dist / front_range[i])
            distance_to_front_normed = sum(min_distances) if len(min_distances) else float('inf')
        else:
            # No pareto yet - assume far from any known good region
            distances_to_front.append(1.0)
            
        
        uncertainty_bonus = 0.
        for name in names:
            std_val = gp_posterior[name]["std"]
            mean_val = gp_posterior[name]["mean"] 
            range_half = front_range[names.index(name)] / 2
            
            # Uncertainty bonus: higher when prediction is far from current Pareto region
            if distance_to_front_normed < float('inf'):
                uncertainty_bonus += std_val * (1.0 - np.exp(-distance_to_front_normed))
        
        values.append(uncertainty_bonus)
    return values