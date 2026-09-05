def score_pool(context):
    """Resamples noisy observations to estimate candidate contribution to Pareto front diversity and hypervolume expansion."""
    n_samples = 30
    names = context["objective_names"]
    
    # Build a set of plausible observation realizations from noise-injected history  
    observed_points = []
    for y in context["Y_obs"]:
        sample_point = np.random.normal(y, 1e-6 * (context["ref_point_by_name"][name] 
                                                  - min(0.0, y[i])) if i < len(names) else 0
                                       for i, name in enumerate(names))
        observed_points.append(sample_point)
    
    # For each candidate: estimate how much it would improve the front under noise  
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            # Add noise to the GP sample
            samples[:,i] = np.random.normal(mean_val, max(std_val, 1e-8), n_samples)
            
        # Estimate how many of these noisy candidates would be non-dominated (and thus useful) 
        dominated_count = 0  
        for i in range(n_samples):
            candidate_sample = samples[i]
            is_dominated = False
            for obs_point in observed_points:
                if all(obs_point[j] >= candidate_sample[j] for j in range(len(names))) and \
                   any(obs_point[j] > candidate_sample[j] for j in range(len(names))):
                    dominated_count += 1 
                    is_dominated = True  
                    break
                    
        # Score based on how many samples would be non-dominated
        score = (n_samples - dominated_count) / n_samples
        
        scores.append(score * cand["acq_value_norm"])
        
    return scores