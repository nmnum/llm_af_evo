def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sample-based dominance checks."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"] 
    n_samples = 20
    lam = 1.0

    scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Draw samples from each objective's posterior distribution  
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in names]
            for _ in range(n_samples)
        ])
        
        improvement_values = []
        
        for s in samples:
            
            # Check if sample is dominated by any point on the current Pareto front
            dominated = False
            
            for q in front:  
                # All components of q >= s AND at least one component > s 
                if np.all(q >= s) and np.any(q > s):
                    dominated = True
                    break
                    
            volume_to_ref = np.prod(np.maximum(s - ref_point, 0))
            
            discounted_volume = volume_to_ref * 0.1 if dominated else volume_to_ref
            
            improvement_values.append(discounted_volume)
        
        mean_improvement = float(np.mean(improvement_values))  
        std_improvement = float(np.std(improvement_values))

        scores.append(mean_improvement - lam * std_improvement)

    return scores