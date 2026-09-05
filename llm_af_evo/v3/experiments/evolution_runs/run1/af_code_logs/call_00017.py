def score_pool(context):
    """Estimates hypervolume improvement using sampled candidates filtered by Pareto dominance probability."""
    n_samples = 20  
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Sample from GP posterior
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in context["objective_names"]]
            for _ in range(n_samples)
        ])

        # Compute probability of being Pareto-optimal via sampling dominance
        dominated_count = 0  
        for s in samples:
            dom_by_any = any(np.all(ref_point >= q) and np.any(q > s) 
                             for q in context["Y_obs"])
            if dom_by_any: 
                dominated_count += 1

        p_pareto = 1.0 - (dominated_count / n_samples)
        
        # Estimate HV improvement using only non-dominated samples
        hv_values = []
        for sample in samples:
            if not any(np.all(ref_point >= q) and np.any(q > sample) 
                       for q in context["Y_obs"]):
                vol = max(0, (sample - ref_point)).prod()
                hv_values.append(vol)
        
        mean_hv_imp = np.mean(hv_values) if hv_values else 0.0
        scores.append(mean_hv_imp * p_pareto)

    return scores