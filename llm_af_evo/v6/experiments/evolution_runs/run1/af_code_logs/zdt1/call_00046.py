def modifier(context):
    """Resample posterior means with noise to estimate candidate contribution to hypervolume expansion and reward those that improve front diversity."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    values = []
    
    # Sample from each candidate's posterior to estimate potential front contribution
    n_samples = 50
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Generate samples of this candidate's objectives 
        sampled_means = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]  
            std_val = gp_posterior[name]["std"]
            
            if not (isinstance(std_val, float) and std_val > 0): 
                sampled_means[:,i] = np.full(n_samples, mean_val)
            else:
                # Draw from normal distribution
                samples = np.random.normal(mean_val, std_val, n_samples)
                sampled_means[:, i] = samples
                
        # Compute hypervolume improvement for each sample relative to current front  
        hv_improvements = []
        
        for s in range(n_samples):
            cand_obj = sampled_means[s]
            
            if not any(np.all(cand_obj >= pt) and np.any(cand_obj > pt)
                       for pt in context["pareto_front"]): 
                # Candidate is dominated, so compute HV contribution from this sample
                hv_improvements.append(0.0)
                
        values.append(max(hv_improvements))  # Use max improvement across samples
    
    return [val * 1e-2 for val in values]