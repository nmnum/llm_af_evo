def modifier(context):
    """Adaptive uncertainty bonus scaled by stagnation and Pareto dominance potential."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"]
    names = context['objective_names']
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    values = []
    n_samples = 50
    base_weight = 0.3
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate Pareto dominance potential via sampling
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        dominated_count = 0
        for sample in samples:
            if any(np.all(sample <= pf_point) and not np.array_equal(sample, pf_point) 
                   for pf_point in pareto_front):
                dominated_count +=1

        p_pareto = 1.0 - (dominated_count / n_samples)
        
        # Uncertainty bonus
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names)

        # Adaptive scaling based on stagnation 
        scale_factor = min(stagnant_batches / 3.0, 1.0) if stagnant_batches > 0 else 1.0
        
        weight = base_weight * scale_factor
        values.append(weight * sigma_norm + p_pareto * 0.2)
        
    return values