def score_pool(context):
    """Exploitation with uncertainty-weighted hypervolume improvement estimation: rank by expected HV gain adjusted for GP confidence."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        
        # Estimate hypervolume improvement using Monte Carlo sampling
        n_samples = 100
        hv_improvement_sum = 0.0
        
        for _ in range(n_samples):
            sample_objectives = np.array([np.random.normal(gp[name]["mean"], gp[name]["std"]) 
                                         for name in names])
            
            # Compute HV contribution of this sample (only if it extends the front)
            if all(sample_objectives >= ref_point):  # dominated by reference point
                hv_improvement_sum += max(0.0, np.prod(ref_point - sample_objectives))
                
        expected_hv_imp = hv_improvement_sum / n_samples
        
        # Weight HV improvement inversely with uncertainty (lower std -> higher score)
        mu_sum = sum(gp[name]["mean"] for name in names) 
        sigma_norm = sum(gp[name]["std"] / context["pareto_front_range"][name] for name in names)

        scores.append(expected_hv_imp * (1.0 + 2.0 * np.exp(-sigma_norm))) 
        
    return scores