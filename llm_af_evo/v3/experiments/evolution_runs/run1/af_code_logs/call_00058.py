def score_pool(context):
    """Scores candidates based on expected hypervolume improvement using Monte Carlo samples from their posteriors, with adaptive uncertainty weighting that decreases over time."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    campaign_progress = context["campaign"]["progress"]

    # Use a fixed number of posterior samples to estimate hypervolume improvement
    n_samples = 50
    
    # Adaptive weight: start with more uncertainty focus early, decrease as we approach the end  
    w_uncert = max(0.1, 0.7 * (1 - campaign_progress))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvement_sum = 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point (normalized to ref_point)
            if all(cand_obj >= ref_point): 
                hv_contrib = 1.0
                for i, name in enumerate(names):
                    hv_contrib *= max(0., ref_point[i] - cand_obj[i]) / front_range[name]
                hv_improvement_sum += hv_contrib
                
        # Normalize by number of samples and apply adaptive weight to uncertainty term  
        score = (hv_improvement_sum/n_samples) + w_uncert * np.mean([gp["f1"]["std"], gp["f2"]["std"]])
        
        scores.append(score)
    
    return scores