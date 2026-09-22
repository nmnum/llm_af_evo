def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws with dominance discounting."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Draw samples from each objective's posterior
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in context["objective_names"]]
            for _ in range(n_samples)
        ])
        
        if len(context["pareto_front"]) == 0:
            # No front yet, use raw hypervolume
            hv_values = np.prod(np.maximum(samples - ref_point, 0), axis=1) 
        else:
            # Check dominance and compute discounted HV values  
           hv_values = []
            for s in samples:
                dominated = False
                for fp in context["pareto_front"]:
                    if all(fp >= s) and any(fp > s):
                        dominated = True
                        break
                        
                hv_raw = np.prod(np.maximum(s - ref_point, 0))
                
                # Discounted HV value  
                discounted_hv =hv_raw * (0.1 if dominated else 1.)
                hv_values.append(discounted_hv)
            hv_values = np.array(hv_values) 
            
        mean_improvement = np.mean(hv_values)
        std_improvement = np.std(hv_values)
        
        score = mean_improvement - lam * std_improvement
        scores.append(score)

    return scores