def modifier(context):
    """Adaptive entropy bonus: penalizes candidates with low uncertainty diversity across objectives, scaled by stagnation."""
    import numpy as np
    
    pool = context["pool"]
    names = context['objective_names']
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Compute entropy of the predictive distribution across objectives
        stds = np.array([gp_posterior[name]["std"] for name in names])
        mean_std = np.mean(stds)
        
        if mean_std > 0: 
            normalized_stds = stds / front_range[names[0]]   # use first obj range as scale reference  
            entropy_bonus = -np.sum(normalized_stds * np.log(normalized_stds + 1e-8))
            
            scaling_factor = min(stagnant_batches / 3.0, 1.0)
            weight = 0.25 * scaling_factor
            
        else:
            # No uncertainty at all
            entropy_bonus = 0.
            weight = 0.

        values.append(weight * entropy_bonus)

    return values