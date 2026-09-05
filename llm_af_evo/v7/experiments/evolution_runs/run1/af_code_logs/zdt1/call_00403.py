def modifier(context):
    """Resample candidate objectives under posterior uncertainty to estimate hypervolume expansion potential."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    values = []
    
    # Use a fixed number of MC samples to estimate the expected hypervolume improvement
    n_samples = 50
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from each objective's posterior distribution (normal)
        sampled_objectives = []
        for _ in range(n_samples):
            sample_vec = [np.random.normal(gp_posterior[name]["mean"], 
                                           max(1e-6, gp_posterior[name]["std"])) 
                          for name in names]
            sampled_objectives.append(sample_vec)

        # For each sample compute hypervolume contribution (relative to ref point)
        hv_contributions = []
        
        for obj_sample in sampled_objectives:
            if all(obj >= rp for obj, rp in zip(obj_sample, ref_point)):
                continue  # skip samples that dominate the reference
            
            contrib_hv = np.prod(np.maximum(0.0, (np.array(ref_point) - np.array(obj_sample))))
            
            hv_contributions.append(contrib_hv)
        
        if not hv_contributions:
            expected_improvement = 0.
        else: 
            # Average over all samples
            expected_improvement = sum(hv_contributions)/len(hv_contributions)

        acq_value_norm = cand["acq_value_norm"]
            
        progress_factor = (1. - context["campaign"]["progress"]) * 2.

        values.append(expected_improvement * acq_value_norm * progress_factor)
    
    return values