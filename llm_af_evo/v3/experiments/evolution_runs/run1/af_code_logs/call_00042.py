def score_pool(context):
    """Estimates hypervolume improvement potential by resampling candidate posteriors and measuring dominance over observed points."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples to estimate hypervolume improvement
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvement_sum  = 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute the HV contribution of this sample point using observed points 
            hypervolume_contribution = 1.0
            for obs_point in X_obs:
                if (cand_obj >= obs_point).all():
                    # This candidate dominates an observation, so we compute its contribution to overall hv  
                    diff_f1 = ref_point[0] - cand_obj[0]
                    diff_f2 = ref_point[1] - cand_obj[1]
                    
                    hypervolume_contribution *= (diff_f1 * diff_f2) if ((cand_obj >= obs_point).all() and not np.array_equal(cand_obj,obs_point)) else 0.0
                elif all(obs_point[i] <= ref_point[i] for i in range(2)): 
                    # The candidate is within the reference box but does not dominate it  
                    hypervolume_contribution *= (ref_point[1]-cand_obj[1]) if cand_obj[1]<obs_point[1] else 0.0
                    
            hv_improvement_sum += max(hypervolume_contribution, 0)
            
        score = hv_improvement_sum / n_samples
        scores.append(score)
    
    return scores