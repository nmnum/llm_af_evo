def score_pool(context):
    """Estimate improvement potential by resampling candidates' objectives from their GPs and computing hypervolume expansion."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV contribution of each candidate via Monte Carlo sampling
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint posterior distribution over objectives (assuming independent Gaussians)
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            if std_val == 0:
                # No uncertainty; just use the point estimate
                samples[:,i] = mean_val  
            else:   
                samples[:, i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Compute hypervolume contribution of each sample (using reference point as lower bound) 
        hv_contributions = []
        for s in samples:
            if all(s[i] <= ref_point[i] for i in range(len(names))):
                contrib = np.prod(ref_point - s)
                hv_contributions.append(contrib)
        
        # Use the mean of hypervolume contributions across MC draws as score
        avg_hv_contribution = 0.0 
        if len(hv_contributions) > 0:
            avg_hv_contribution = sum(hv_contributions)/len(hv_contributions)

        scores.append(avg_hv_contribution)
    
    return scores