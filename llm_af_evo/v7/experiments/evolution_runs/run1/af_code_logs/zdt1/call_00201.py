def modifier(context):
    """Estimates candidate contribution to front coverage gap by sampling noisy posterior means and measuring how often they extend under-covered regions."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    n_samples = 25
    values = []
    
    # For each candidate, sample noisy means from GP posteriors and estimate front expansion potential  
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample objective vectors under noise (as if we observed them)
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Compute hypervolume contribution of each sample (relative to ref point)
        hv_contributions = []
        for s in samples:
            if all(s >= ref_point):
                vol = 1.0
                for i, val in enumerate(s - ref_point): 
                    vol *= max(0., val)  
                hv_contributions.append(vol)

        # Estimate expected hypervolume improvement from this candidate's noisy posterior sample
        exp_hv_improvement = np.mean(hv_contributions)
        
        values.append(exp_hv_improvement * cand["acq_value_norm"])
    
    return values