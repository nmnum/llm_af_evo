def modifier(context):
    """Adaptive uncertainty bonus scaled by how close candidates are to current pareto front in objective space."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    pareto_front = context["pareto_front"]
    
    if len(pareto_front) == 0:
        return [0.0] * len(pool)
        
    names = context['objective_names']
    front_range = context["pareto_front_range"]

    # Compute distance from each candidate to the nearest point on Pareto front
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        mean_vec = np.array([gp_posterior[name]["mean"] for name in names])
        
        distances_to_pf = [np.linalg.norm(mean_vec - pf_point) / 
                          np.sqrt(sum((front_range[n] ** 2) for n in names))
                           for pf_point in pareto_front]
            
        min_distance_norm = min(distances_to_pf)
                
        # Uncertainty bonus
        sigma_sum = sum(gp_posterior[name]["std"] for name in names)

        # Scale uncertainty bonus based on proximity to front: less uncertain if near the frontier 
        weight = 0.3 * (1 - np.clip(min_distance_norm, 0, 1))
        
        values.append(weight * sigma_sum)
    
    return values