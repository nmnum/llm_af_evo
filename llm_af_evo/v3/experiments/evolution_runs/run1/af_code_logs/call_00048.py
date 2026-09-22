def score_pool(context):
    """Estimates hypervolume improvement potential by resampling candidate objectives under noise, weighted by proximity to existing observations."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvement_sum = 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point relative to current front and ref_point 
            if all(cand_obj >= pf) or not any((cand_obj <= pf).all() for pf in context["pareto_front"]):  
                hv_improvement_sum += max(0, np.prod(ref_point - cand_obj))
        
        # Normalize by number of samples
        score = 1.0 * hv_improvement_sum / n_samples
        
        scores.append(score)
    
    return scores