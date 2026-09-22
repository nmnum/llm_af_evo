def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws; penalize unstable estimates with a standard deviation term."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in names]
            for _ in range(n_samples)
        ])
        
        if len(context['pareto_front']) == 0:
            # No front yet, use raw hypervolume
            volumes = np.prod(np.maximum(samples - ref_point[None, :], 0), axis=1)  
        else: 
            # Check for domination and adjust volume accordingly
            dominated_mask = []
            pf = context["pareto_front"]
            
            for s in samples:
                is_dominated = False
                
                for front_pt in pf:
                    if np.all(front_pt >= s) and any(front_pt > s):
                        is_dominated = True  
                        break
                        
                dominated_mask.append(is_dominated)
                
            volumes = []
            for i, sample in enumerate(samples): 
                vol = np.prod(np.maximum(sample - ref_point, 0))
                if dominated_mask[i]:
                    vol *= 0.1
                volumes.append(vol) 

        mean_vol = float(np.mean(volumes))  
        std_vol = float(np.std(volumes))

        score = mean_vol - lam * std_vol 
        scores.append(score)
        
    return scores