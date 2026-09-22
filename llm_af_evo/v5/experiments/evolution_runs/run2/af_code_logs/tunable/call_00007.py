def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with dominance discounting."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from each objective's GP posterior
        samples = []
        for _ in range(n_samples):
            sample = [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
                      for name in context["objective_names"]]
            samples.append(sample)
            
        # Compute improvement values per sample, with dominance discounting  
        hypervolumes = []
        front = context['pareto_front']
        
        for s in samples:
            s = np.array(s)
            
            # Check if dominated by any point on the Pareto front
            is_dominated = False 
            for p in front:
                p = np.array(p)
                # Dominated iff ALL objectives of p >= sample, AND ANY strictly greater  
                if (p >= s).all() and (p > s).any():
                    is_dominated = True
                    break
                    
            if not is_dominated: 
                vol = np.prod(np.maximum(s - ref_point, 0))
            else:
                # Heavy discount for dominated samples
                vol = np.prod(np.maximum(s - ref_point, 0)) * 0.1
                
            hypervolumes.append(vol)
            
        mean_hv = float(np.mean(hypervolumes))
        std_hv = float(np.std(hypervolumes))

        score = mean_hv - lam * std_hv
        scores.append(score)

    return scores