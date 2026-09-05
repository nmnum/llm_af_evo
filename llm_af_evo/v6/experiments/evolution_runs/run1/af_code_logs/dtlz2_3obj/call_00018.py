def modifier(context):
    """Estimate candidate expansion potential of dominated points via noisy posterior sampling."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    names = context['objective_names']
    
    # Use a fixed noise level for Monte Carlo estimation  
    noise_level = 0.15
    n_samples = 20 
    
    values = []
    
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the candidate's posterior and estimate hypervolume expansion potential 
        samples = np.zeros((n_samples, len(names)))
        dominated_count = 0
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]  
            std_val = gp_posterior[name]["std"]
            
            # Add noise to simulate uncertainty
            sample_vals = np.random.normal(mean_val, max(std_val, 1e-6), n_samples)
            samples[:,i] = sample_vals
            
        ref_point = context["ref_point"]

        # Check if each sampled point dominates any existing observation  
        for s in samples:
            dom = False 
            for obs_y in Y_obs:   
                if all(s[i] >= obs_y[i] and not (s == obs_y).all() for i in range(len(names))):
                    dom = True
                    break
                    
            # If point is dominated by current observations, it may still improve front  
            if dom:
                dominated_count += 1
                
        p_improvement = 1.0 - (dominated_count / n_samples)
        
        values.append(p_improvement * noise_level)

    return values