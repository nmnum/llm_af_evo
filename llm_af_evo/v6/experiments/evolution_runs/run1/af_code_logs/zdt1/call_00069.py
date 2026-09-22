def modifier(context):
    """Exploitation-uncertainty trade-off biasing toward recent improvement trends and penalizing redundant uncertainty."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    names = context['objective_names']
    
    # Track the trend of objective improvements over time
    n_obs = len(Y_obs)
    if n_obs < 2:
        return [0.0] * len(pool) 
    
    recent_improvements = []
    for i in range(len(names)):
        y_series = Y_obs[:,i]
        diff = np.diff(y_series[-3:]) 
        avg_diff = np.mean(diff)  
        recent_improvements.append(avg_diff)
        
    # Determine if we're improving, stagnating or regressing on average
    improvement_trend = sum(1 for x in recent_improvements if x > 0.02) / len(names)

    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Compute uncertainty as the weighted std of objectives 
        sigma_norm = np.mean([gp_posterior[name]["std"] for name in names])
            
        if improvement_trend > 0.5:  
            # We're improving, prefer exploitation (lower uncertainty bonus)
            weight = -sigma_norm * 0.1
        else:
            # Stagnation or regression — increase exploration 
            base_weight = sigma_norm * 0.3
            
            # Reduce the effect of this correction if candidate already has high acquisition value  
            acq_val = cand["acq_value_norm"]    
            
            weight = max(0, (1 - acq_val) ** 2 ) * base_weight 

        values.append(weight)
        
    return values