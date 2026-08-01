def score_pool(context):
    """Estimate each candidate’s potential hypervolume improvement by resampling objectives from their GP posteriors, then rank by expected HV gain adjusted for proximity to existing observations."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Prepare observation history
    X_obs = context["X_obs"] 
    Y_obs = context["Y_obs"]

    scores = []
    n_samples = 50
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample objectives from candidate's GP posterior  
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Compute hypervolume contribution of each sample
        hv_contributions = []
        for s in samples:
            if all(s[i] <= ref_point[i] for i in range(len(names))):  # feasible?
                contrib = max(0.0, np.prod(ref_point - s))
                hv_contributions.append(contrib)
        
        expected_hv_improvement = np.mean(hv_contributions) if hv_contributions else 0.0
        
        scores.append(expected_hv_improvement)

    return scores