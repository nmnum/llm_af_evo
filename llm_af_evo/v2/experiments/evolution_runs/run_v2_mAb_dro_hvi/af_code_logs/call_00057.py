def score_pool(context):
    """Estimate improvement in hypervolume by resampling candidate predictions under uncertainty, then rank based on expected HV gain."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Resample each candidate's GP posteriors to estimate HV contribution
    n_samples = 50
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample predictions from the posterior distributions of all objectives
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Estimate hypervolume improvement by comparing with current reference point
        hv_gains = []
        for sample in samples:
            if all(sample >= ref_point):  # All objectives are better than the reference (dominated region is expanded)
                continue
            
            # Compute HV contribution of this sampled objective vector vs. existing front and ref point  
            hypervolume_contribution = np.prod(ref_point - np.maximum(sample, ref_point))
            
            hv_gains.append(hypervolume_contribution)

        expected_hv_gain = (np.mean(np.array(hv_gains)) if len(hv_gains) > 0 else 0.0)
        
        scores.append(expected_hv_gain)

    return scores