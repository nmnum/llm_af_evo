def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    import numpy as np
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point"]
    
    # Early stage: more exploitation, later: more exploration
    progress_factor = 0.5 + 0.5 * (context['campaign']['progress'])
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        acq_val_norm = cand["acq_value_norm"]
        
        # Exploitation: weighted mean of objective means
        mu_sum = sum(gp_posterior[name]["mean"] for name in names)
        exploitation_score = 0.5 * (mu_sum / len(names))
        
        # Uncertainty: normalized std across objectives  
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] 
                         for name in names) / len(names)
         
        # Novelty: inverse distance to nearest observed point
        cand_x = np.array(cand["x"])
        if context['X_obs'].size > 0:
            distances = np.linalg.norm(context['X_obs'] - cand_x, axis=1)
            novelty_score = 1.0 / (np.min(distances) + 1e-8)
        else: 
            novelty_score = 1.0
            
        # Combine components with dynamic weights based on progress
        w_exploit = 0.3 * progress_factor  
        w_uncert = 0.4 * (1 - progress_factor)   
        w_novelty = 0.3
        
        score = (
            acq_val_norm + 
            w_exploit * exploitation_score +
            w_uncert * sigma_norm + 
            w_novelty * novelty_score
        )
        
        scores.append(score)
    
    return scores