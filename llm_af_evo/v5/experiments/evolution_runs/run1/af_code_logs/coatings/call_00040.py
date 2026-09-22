def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with dominance discounting."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"] 
    n_samples = 20
    lam = 1.0

    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior  
        samples = np.zeros((n_samples, len(names)))
        rng = np.random.default_rng()
        for i, name in enumerate(names):
            mean = gp[name]["mean"] 
            std = gp[name]["std"]
            samples[:,i] = rng.normal(mean, std, n_samples)
        
        # Compute improvement values per sample
        vol_improvements = []
        for s in samples:
            
            # Check if dominated by any front point  
            is_dominated = False
            for f in front:
                dominates = np.all(f >= s) and np.any(f > s)
                if dominates: 
                    is_dominated = True
                    break
            
            vol = np.prod(np.maximum(s - ref_point, 0))
            
            # Discount dominated samples heavily  
            if is_dominated:
                vol *= 0.1
                
            vol_improvements.append(vol) 
            
        mean_imp = float(np.mean(vol_improvements)) 
        std_imp = float(np.std(vol_improvements))

        score = mean_imp - lam * std_imp
        scores.append(score)
        
    return scores