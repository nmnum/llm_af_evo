def modifier(context):
    """Boost candidates that are diverse in objective space and away from current non-dominated points, using a resampling approach to estimate potential dominance."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    pareto_front = context["pareto_front"]
    
    # Estimate probability of being Pareto-optimal via noisy posterior sampling
    n_samples = 100
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]

        # Sample from the joint GP distribution (assuming independent objectives for simplicity)
        samples = np.zeros((n_samples, len(context['objective_names'])))
        
        for i, name in enumerate(context['objective_names']):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            # Sample from normal distribution (this is a simplified approximation)
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        dominated_count = 0
        for sample in samples:
            if any(np.all(sample <= pf_point) and not np.array_equal(sample, pf_point) 
                   for pf_point in pareto_front):
                # This sampled point is dominated by at least one Pareto front member  
                dominated_count +=1

        p_pareto = 1.0 - (dominated_count / n_samples)
        
        # Add a bonus if the candidate has potential to be on or near the PF
        values.append(p_pareto * 0.2) 
        
    return values