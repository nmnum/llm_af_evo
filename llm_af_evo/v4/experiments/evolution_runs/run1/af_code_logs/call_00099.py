def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws with dominance-aware discounting."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Draw samples from the joint posterior distribution of objectives.
        means = [gp_posterior[name]["mean"] for name in ["f1", "f2"]]
        stds = [gp_posterior[name]["std"] for name in ["f1", "f2"]]
        samples = np.random.normal(loc=means, scale=stds, size=(n_samples, 2))
        
        # Compute hypervolume improvement values per sample.
        vol_improvements = []
        front = context["pareto_front"]
        if len(front) == 0:
            for s in samples:
                volume = np.prod(np.maximum(s - ref_point, 0))  
                vol_improvements.append(volume)
        else: 
            # Check dominance of each sample.
            for s in samples:
                is_dominated = False
                for q in front:
                    if all(q[i] >= s[i] for i in range(2)) and any(q[i] > s[i] for i in range(2)):
                        is_dominated = True  
                        break
                
                volume = np.prod(np.maximum(s - ref_point, 0))
                
                # Discount dominated samples.
                if is_dominated:
                    vol_improvements.append(volume * 0.1)
                else: 
                    vol_improvements.append(volume)

        mean_imp = float(np.mean(vol_improvements))  
        std_imp = float(np.std(vol_improvements))

        score = mean_imp - lam * std_imp
        scores.append(score) 
    
    return scores