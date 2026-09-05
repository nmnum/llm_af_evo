def modifier(context):
    """Adaptive uncertainty bonus scaled by stagnation and proximity-based suppression of near-duplicates."""
    import numpy as np
    
    pool = context["pool"]
    X_obs = context['X_obs']
    names = context['objective_names'] 
    front_range = context["pareto_front_range"]
    
    stagnant_batches = context["campaign"]["stagnant_batches"]  
    base_weight = 0.3
    scaling_factor = min(stagnant_batches / 5.0, 1.0)
    weight = base_weight * scaling_factor
    
    values = []
    
    for i, cand in enumerate(pool):
        gp_posterior = cand["gp_posterior"]
        
        # Compute uncertainty bonus  
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
        
        # Proximity-based suppression: if candidate is very close to any observed point,
        # reduce its correction term
        cand_x = np.array(cand["x"])
        min_dist_to_observed = float('inf')
        
        for obs_x in X_obs:
            dist = np.linalg.norm(cand_x - obs_x)
            min_dist_to_observed = min(min_dist_to_observed, dist)

        # Suppress correction if candidate is too close to existing observations
        suppression_factor = 1.0  
        if min_dist_to_observed < 0.05:   # threshold can be tuned 
             suppression_factor *= (min_dist_to_observed / 0.05) ** 2
            
        values.append(weight * sigma_norm * suppression_factor)

    return values