def score_pool(context):
    """Estimates probability that a candidate expands the pareto front using noisy GP samples, favoring those with highest expected HV increase."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"] 
    pf = context["pareto_front"]

    scores = []
    
    # Sample from each cand's posterior to estimate hypervolume improvement
    n_samples = 20  
    hv_improvements = []

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Generate samples of objectives 
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_, std_ = gp[name]["mean"], gp[name]["std"]  
            samples[:,i] = np.random.normal(mean_, std_, n_samples)
            
        # Estimate HV improvement by sampling
        cand_hv_improvements = []
        
        for s in samples:
            new_pf = np.vstack([pf, s])
            # Compute hypervolume of the new front (assuming no dominated points)  
            hv_new = 1.0 
            for i, name in enumerate(names):
                hv_new *= max(0., ref_point[i] - min(new_pf[:,i]))
                
           hv_old = 1.0
            for i, name in enumerate(names):   
                hv_old *= max(0., ref_point[i] - min(pf[:,i])) 
            
            cand_hv_improvements.append(max(0., hv_new - hv_old))
            
        # Expected HV improvement 
        expected_hv_imp = np.mean(cand_hv_improvements)
        
        scores.append(expected_hv_imp)

    return scores