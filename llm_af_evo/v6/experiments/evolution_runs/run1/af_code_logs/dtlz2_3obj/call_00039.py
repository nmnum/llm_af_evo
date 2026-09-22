def modifier(context):
    """Adaptive entropy bonus based on GP posterior uncertainty distribution shape."""
    import numpy as np
    
    pool = context["pool"]
    names = context['objective_names']
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Adaptive weight scaling with stagnation
    base_weight = 0.3  
    scaling_factor = min(stagnant_batches / 5.0, 1.0)
    weight = base_weight * scaling_factor
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Compute entropy-like uncertainty metric from GP stds
        sigmas = [gp_posterior[name]["std"] for name in names]
        normalized_sigmas = np.array(sigmas) / np.array([front_range[name] for name in names])
        
        # Use log-sum-exp to emphasize larger uncertainties while remaining differentiable  
        entropy_bonus = weight * (np.log(np.sum(np.exp(normalized_sigmas))) - np.log(len(names)))
            
        values.append(entropy_bonus)
    
    return values