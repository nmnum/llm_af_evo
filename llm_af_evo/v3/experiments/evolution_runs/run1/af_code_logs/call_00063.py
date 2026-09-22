def score_pool(context):
    """Scores candidates based on expected hypervolume improvement using sampled objective values, with progress-aware exploitation vs uncertainty trade-off."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"] 
    campaign_progress = context["campaign"]["progress"]

    # Use a fixed number of posterior samples to estimate improvement
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        total_hv_improvement = 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point relative to current front 
            hv_contrib = max(0., (ref_point[0] - cand_obj[0]) * (ref_point[1] - cand_obj[1]))
            
            total_hv_improvement += hv_contrib

        score = 1.0 * total_hv_improvement / n_samples
        
        # Adjust balance between exploitation and uncertainty based on progress
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names) 
                
        weight_exploit = 0.5 + 0.5 * campaign_progress # shift from exploration to exploitation  
        score_final = (1 - weight_exploit) * sigma_norm + weight_exploit * mu_sum
        
        scores.append(score_final)
    
    return scores