def score_pool(context):
    """Resamples noisy Pareto front estimates to assess candidate robustness and hypervolume expansion potential."""
    names = context["objective_names"]
    ref_point = context["ref_point"] 
    X_obs = context["X_obs"]
    
    # Estimate noise level from observation history  
    if len(X_obs) > 1:
        x_std = np.std(X_obs, axis=0)
        noise_scale = max(np.mean(x_std), 1e-6)
    else:   
        noise_scale = 1.0
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample candidate's objectives multiple times under noise
        n_samples = min(20, max(5, len(X_obs) // 2))
        sampled_hv_gains = [] 
        
        for _ in range(n_samples):
            # Perturb the candidate features slightly 
            x_perturbed = cand["x"] + np.random.normal(scale=noise_scale/10.0, size=cand["x"].shape)
            
            # Estimate hypervolume gain from this perturbation
            hv_gain = 0.
            for _ in range(5):   # Few samples per candidate to keep cost low  
                y_sampled = []
                for name in names:
                    mean_val = gp[name]["mean"] 
                    std_val = gp[name]["std"]
                    
                    if np.random.rand() < 0.1:    # Occasionally sample from noise
                        sampled_y = mean_val + std_val * np.random.randn()
                    else:
                        sampled_y = max(mean_val - 2*std_val, min(mean_val + 2*std_val, 
                                                                 np.random.normal(loc=mean_val,scale=std_val)))
                    
                    y_sampled.append(sampled_y)
                
                # Compute hypervolume contribution of this sample
                if all(y >= ref_point[i] for i,y in enumerate(y_sampled)):
                    hv_gain += 1.0   # Placeholder: could be replaced with actual HV computation 
            sampled_hv_gains.append(hv_gain) 
            
        avg_hv_improvement = np.mean(sampled_hv_gains)
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        scores.append(avg_hv_improvement + 0.5 * mu_sum)

    return scores