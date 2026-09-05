def score_pool(context):
    """Estimate candidate domination risk by perturbing objectives under uncertainty and scoring based on robustness."""
    names = context["objective_names"]
    scores = []
    
    # Get reference point for HV calculation  
    ref_point = np.array([context['ref_point_by_name'][name] for name in names])
    
    # For each candidate, sample noisy versions of its objectives from the GP posterior
    n_samples = 100
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample objective values under noise (assumes Gaussian posteriors)
        samples_f1 = np.random.normal(gp_posterior['f1']['mean'], gp_posterior['f1']['std'], n_samples) 
        samples_f2 = np.random.normal(gp_posterior['f2']['mean'], gp_posterior['f2']['std'], n_samples)

        # Count how many times this candidate dominates the reference point (i.e., is better than ref in all objectives)
        dominated_count = 0
        for f1_samp, f2_samp in zip(samples_f1, samples_f2):
            if np.all([f1_samp >= ref_point[0], f2_samp >= ref_point[1]]):  
                dominated_count += 1
                
        # Score is the proportion of samples that dominate reference (higher = more promising)
        score = dominated_count / n_samples
        
        scores.append(score)

    return scores