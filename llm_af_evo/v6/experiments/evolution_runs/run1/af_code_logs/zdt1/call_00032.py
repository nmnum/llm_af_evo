def modifier(context):
    """Adaptive uncertainty bonus combined with Pareto dominance potential scaled by stagnation."""
    import numpy as np
    
    pool = context["pool"]
    pareto_front = context["pareto_front"] 
    names = context['objective_names']
    front_range = context["pareto_front_range"]
    
    stagnant_batches = context["campaign"]["stagnant_batches"]  
    base_weight = 0.3
    scaling_factor = min(stagnant_batches / 5.0, 1.0)
    weight = base_weight * scaling_factor
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate dominance potential via sampling 
        n_samples = 20
        samples = np.zeros((n_samples, len(names)))
        dominated_count = 0
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]  
            std_val = gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            
        # Check if sample is dominated by any point on the Pareto front
        for s in samples:
            dom = False 
            for pf_point in pareto_front:
                if all(s[i] <= pf_point[i] and not (s == pf_point).all() for i in range(len(names))):
                    dom = True  
                    break
            if dom: dominated_count += 1
            
        p_pareto = 1.0 - (dominated_count / n_samples)
        
        # Combine uncertainty with dominance potential 
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
        
        values.append(weight * sigma_norm + p_pareto * 0.2)

    return values