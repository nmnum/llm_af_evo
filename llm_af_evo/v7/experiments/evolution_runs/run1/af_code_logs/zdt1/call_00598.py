def modifier(context):
    """Noise-resampled dominance bonus: estimate how often a candidate would dominate after noise perturbation."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Number of resamples for Monte Carlo estimation
    n_resample = 20

    values = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Collect means and stds from GP posterior (already oriented to be higher is better)
        mean_vals = np.array([gp_posterior[name]["mean"] for name in names])
        std_vals = np.array([gp_posterior[name]["std"] for name in names])

        # Generate noisy samples of the candidate's objectives
        candidates_noisy_samples = []
        
        if n_resample > 0:
            noise_sampled = np.random.normal(mean_vals, std_vals, (n_resample, len(names)))
            
            # Count how many times this sample dominates a point in pareto_front 
            dominance_count = 0
            
            for noisy_obj_val in noise_sampled:  
                is_dominant = True
                for pf_point in context["pareto_front"]:
                    if all(noisy_obj_val <= pf_point):
                        is_dominant = False
                        break
                        
                if is_dominant:
                    dominance_count += 1
                    
            # Estimate probability of domination, scaled by acquisition value 
            prob_dominate = float(dominance_count) / n_resample
            
        else:  
             prob_dominate = 0.0

        
         values.append(3 * (prob_dominate)) 
        
    return values