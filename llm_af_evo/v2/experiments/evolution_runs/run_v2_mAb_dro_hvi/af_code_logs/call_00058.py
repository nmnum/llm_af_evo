def score_pool(context):
    """Exploitation with hypervolume-improvement estimation: rank by predicted objective sum adjusted for estimated marginal gain in dominated volume."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV improvement using Monte Carlo sampling from GP posteriors
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objectives for this candidate multiple times to estimate uncertainty
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Calculate hypervolume improvement over current Pareto front
        cand_hv_improvement = 0.0
        for sample in samples:
            if all(sample[i] >= ref_point[i] or i == len(names)-1 
                   for i in range(len(names))): continue
            
            # Check how much this point would expand the dominated volume (simplified)
            hv_gain_per_sample = np.prod(np.maximum(ref_point - sample, 0))
            cand_hv_improvement += hv_gain_per_sample
        
        avg_hv_imp = cand_hv_improvement / n_samples
        mu_sum = sum(gp[name]["mean"] for name in names)

        # Combine exploitation with HV estimate (higher is better)
        scores.append(mu_sum + 0.5 * avg_hv_imp) 

    return scores