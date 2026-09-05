def score_pool(context):
    """Estimates hypervolume improvement by resampling predicted objectives and computing probability of non-domination."""
    n_samples = 30
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Sample from GP posterior per objective  
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in context["objective_names"]]
            for _ in range(n_samples)
        ])

        # Compute how many sampled points are non-dominated by current front
        n_non_dom = 0
        if len(context["pareto_front"]) == 0:
            # No existing frontier, all samples contribute to hypervolume  
            scores.append(np.mean(np.prod(np.maximum(samples - ref_point, 0), axis=1)))
            continue

        for s in samples:
            dominated = False 
            for q in context["pareto_front"]:
                if np.all(q >= s) and np.any(q > s):
                    dominated = True
                    break
            
            # If not dominated by current front, compute contribution to HV  
            if not dominated:   
                n_non_dom += 1
        
        prob_nondom = float(n_non_dom)/n_samples 
        scores.append(prob_nondom)
    
    return scores