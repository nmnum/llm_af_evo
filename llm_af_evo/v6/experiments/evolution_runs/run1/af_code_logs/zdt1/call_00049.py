def modifier(context):
    """Penalty for candidates with low uncertainty-weighted objective means to encourage exploration of promising regions."""
    import numpy as np
    
    pool = context["pool"]
    names = context['objective_names']
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Compute weighted mean (higher is better, already flipped)
        weight_sum = 0.0
       weighted_mean_sum = 0.0
        
        for name in names: 
            mean_val = gp_posterior[name]["mean"]  
            std_val = gp_posterior[name]["std"]
            
            # Use inverse of standard deviation as weighting (more certain -> higher weight)
            if std_val > 1e-8:
                w = 1.0 / std_val
            else: 
                w = 1.0
                
            weighted_mean_sum += mean_val * w  
            weight_sum += w
            
        # Avoid division by zero for all-zero weights (shouldn't happen in practice)
        if weight_sum > 0:
            normalized_weighted_mean = weighted_mean_sum / weight_sum
        else: 
            normalized_weighted_mean = sum(gp_posterior[name]["mean"] for name in names) 

        # Normalize the mean using front range  
        norm_mean = sum((normalized_weighted_mean - gp_posterior[name]["mean"]) / front_range[name]  if front_range[name] > 0. else 0. 
                        for name in names)
        
        # Apply a penalty based on how low this weighted normalized mean is
        scaling_factor = min(stagnant_batches / 3.0, 1.0)  
        correction = -0.2 * norm_mean * scaling_factor
        
        values.append(correction)

    return values