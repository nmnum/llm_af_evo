def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws; penalize unstable predictions."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0

    names = context["objective_names"]
    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"]

    scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        if len(pareto_front) == 0:
            # No front yet, so all points contribute full volume
            improvement_values = np.prod(np.maximum(samples - ref_point, 0), axis=1)
        else:
            # Check domination and adjust volumes accordingly 
            improved_volumes = []
            
            for s in samples:  
                dominated = False
                
                for pf_point in pareto_front:
                    if all(pf_point >= s) and any(pf_point > s):
                        dominated = True
                        break
                        
                vol_s = np.prod(np.maximum(s - ref_point, 0))
                
                improved_volumes.append(vol_s * (1.0 if not dominated else 0.1)) 
                    
            improvement_values = np.array(improved_volumes)
            
        mean_imp = float(np.mean(improvement_values))
        std_imp = float(np.std(improvement_values))

        scores.append(mean_imp - lam * std_imp)

    return scores