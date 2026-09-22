def score_pool(context):
    """Estimate probabilistic pareto-optimality by sampling noisy objectives and compute expected hv-improvement."""
    names = context["objective_names"]
    ref_point = np.array(context["ref_point"])
    
    # Sample from each candidate's posterior to estimate hypervolume improvement
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Draw samples per objective (mean ± std * N(0,1))
        f1_mean, f1_std = gp_posterior['f1']['mean'], gp_posterior['f1']['std']
        f2_mean, f2_std = gp_posterior['f2']['mean'], gp_posterior['f2']['std']

        samples_f1 = np.random.normal(f1_mean, f1_std, n_samples)
        samples_f2 = np.random.normal(f2_mean, f2_std, n_samples)

        # Combine into sample points
        sampled_points = np.column_stack((samples_f1, samples_f2))

        # Estimate probability that this candidate expands the hypervolume:
        # how many of its noisy samples are dominated by current front?
        
        hv_improvement_estimate = 0.0
        
        for point in sampled_points:

            if all(point >= ref_point): 
                continue
            
            is_dominated_by_front = False
            for pf_point in context["pareto_front"]:
                
                # Check domination: a <= b element-wise and not identical  
                dominated = np.all(pf_point >= point) and not np.array_equal(pf_point, point)
                if dominated:
                    is_dominated_by_front = True 
                    break

            if not is_dominated_by_front:

                 hv_improvement_estimate += 1.0 

        # Normalize by number of samples to get probability
        prob_not_dominated = hv_improvement_estimate / n_samples
        
        scores.append(prob_not_dominated)

    return scores