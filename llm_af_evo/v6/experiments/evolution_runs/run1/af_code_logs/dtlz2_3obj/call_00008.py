def modifier(context):
    """Combine uncertainty bonus with Pareto dominance potential, scaled by stagnation level for adaptive exploration."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    pareto_front = context["pareto_front"]
    
    # Estimate probability of being Pareto-optimal via noisy posterior sampling
    n_samples = 100
    names = context['objective_names']
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"] 
    base_weight = 0.3
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]

        # Sample from the joint GP distribution (assuming independent objectives)
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        dominated_count = 0
        for sample in samples:
            if any(np.all(sample <= pf_point) and not np.array_equal(sample, pf_point) 
                   for pf_point in pareto_front):
                # This sampled point is dominated by at least one Pareto front member  
                dominated_count +=1

        p_pareto = 1.0 - (dominated_count / n_samples)
        
        # Combine uncertainty and dominance potential
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
        scaling_factor = min(stagnant_batches / 5.0, 1.0)
        weight = base_weight * scaling_factor
        
        values.append((weight * sigma_norm + p_pareto * 0.2))
        
    return values