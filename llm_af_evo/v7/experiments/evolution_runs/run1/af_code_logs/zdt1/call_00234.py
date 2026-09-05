def modifier(context):
    """Resamples posterior means with added Gaussian noise to estimate robustness of candidate domination and rewards those that are less sensitive to perturbations."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Early exit if no observations yet 
    if len(context.get("X_obs", [])) == 0:
        return [0.] * len(context["pool"])
        
    n_samples = 128
    values = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        means = np.array([gp_posterior[name]["mean"] for name in names])
        stds = np.array([gp_posterior[name]["std"] for name in names])

        # Add noise to the mean and check how often it changes dominance status
        dominated_counts = []
        
        for _ in range(n_samples):
            noisy_means = means + np.random.normal(0, 1e-8, len(means)) 
            candidate_point = noisy_means
            
            is_dominated = False  
            
            # Check if this perturbed point would be dominated by current front
            pf_points = context["pareto_front"]
        
            for pf in pf_points:
                if all(pf >= candidate_point):
                    is_dominated = True
                    break
                    
            dominated_counts.append(1.0 if is_dominated else 0.0)
            
        # Fraction of perturbed samples that still dominate the front  
        frac_domination_robustness = np.mean(dominated_counts) 
        
        # Reward candidates with lower robustness (i.e., more likely to stay non-dominated after noise).
        score = -frac_domination_robustness
        
        values.append(score)
        
    max_val = np.max(values)
    
    if max_val > 0:
         return [v / max_val for v in values]
    else: 
         return [float(v) for v in values]