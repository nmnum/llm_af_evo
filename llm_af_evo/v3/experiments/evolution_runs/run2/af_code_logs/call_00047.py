def score_pool(context):
    """Estimates hypervolume improvement by sampling noisy predictions and measuring expansion potential."""
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"] 
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample noisy predictions from GP posteriors
        n_samples = 100
        samples = [] 
        for _ in range(n_samples):
            sample_y = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            samples.append(sample_y)
        
        # Compute hypervolume improvement potential using sampled points  
        cand_hv_improvement = 0.0
        ref_point_vec = np.array(ref_point) 
        for sample in samples:
            if all(s >= r for s, r in zip(sample, ref_point)):
                continue  # Sample is dominated by reference
            
            # Estimate hypervolume expansion: how much more area would be added?
            expanded_ref = [max(r, s) for r, s in zip(ref_point_vec, sample)]
            
            hv_expansion = np.prod(expanded_ref - ref_point)
            cand_hv_improvement += hv_expansion
            
        # Normalize by number of samples
        avg_hv_impact = cand_hv_improvement / n_samples
        
        scores.append(avg_hv_impact)

    return scores