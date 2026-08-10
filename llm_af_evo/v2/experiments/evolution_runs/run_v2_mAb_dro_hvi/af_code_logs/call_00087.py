def score_pool(context):
    """Estimate improvement potential by resampling candidates' posteriors and measuring how often they dominate or are dominated; rewards those that could shift the Pareto front."""
    import numpy as np
    
    names = context["objective_names"]
    n_samples = 50
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        # Sample candidate's objectives from its GP posterior
        samples = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
        
        domination_count = 0  
        dominated_by_count = 0
        
        # For each observed point, check if this sample dominates or is dominated 
        for obs_point in context["Y_obs"]:
            # Sample is better than observation
            if all(samples[i] >= obs_point[i] for i in range(len(names))):
                domination_count += 1
                
            # Observation is better than sample  
            elif all(obs_point[i] >= samples[i] for i in range(len(names))):
                dominated_by_count += 1
        
        # Score based on how often it could improve the front
        score = (domination_count - dominated_by_count) / n_samples if n_samples > 0 else 0.0  
        
        scores.append(score)
    
    return scores