def modifier(context):
    """Adaptive uncertainty bonus biased toward under-explored regions of objective space."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    names = context['objective_names']
    
    # Compute the normalized variance across objectives for each candidate
    values = []
    front_range = context["pareto_front_range"]

    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate uncertainty as mean of standardized stds 
        sigma_norm = np.mean([gp_posterior[name]["std"] / front_range[name]  for name in names])
         
        # Adaptive scaling based on how underexplored the candidate's objective region is
        if len(Y_obs) > 0:
            # Compute distance to nearest observed point (in normalized space)
            cand_x = np.array(cand["x"])
            
            distances = [np.linalg.norm(cand_x - obs_x, ord=2) for obs_x in context['X_obs']]
        
            min_distance = min(distances)

            if min_distance > 0.1: # If not too close to existing points
                weight_factor = sigma_norm * (1 / max(min_distance , 0.05)) 
            else:
               weight_factor = -sigma_norm * 0.2  
        else:
           weight_factor = sigma_norm
            
        
        values.append(weight_factor)
    return values