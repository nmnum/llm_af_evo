def score_pool(context):
    """Estimates improvement potential by resampling candidates' objectives from their posteriors and scoring based on hypervolume contribution."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use a small Monte Carlo sample to estimate HV improvement
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample objectives from the candidate's GP posterior
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else:
                samples[:,i] = mean_val
        
        # Score based on how much each sample contributes to HV
        hv_contributions = []
        for s in samples[:10]:  # Use a subset of samples per candidate for efficiency  
            if np.all(s <= ref_point): 
                contrib_hv = max(0, (ref_point - s).prod())
                hv_contributions.append(contrib_hv)
        
        score = sum(hv_contributions) / len(samples[:10]) if hv_contributions else 0
        scores.append(score)

    return scores