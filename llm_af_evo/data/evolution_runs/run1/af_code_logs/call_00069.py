def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation: mean of objectives
        mu_sum = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean'])
        
        # Uncertainty: normalized standard deviation
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + 
                     gp['kD']['std']/front_range['kD'] + 
                     gp['viscosity']['std']/front_range['viscosity'])
        
        # Novelty: distance to nearest observed point
        dist_to_obs = float(np.linalg.norm(X_obs - cand["x"], axis=1).min())
        
        # Adaptive weighting based on progress and stagnation
        exploit_weight = max(0.5, 1.0 - progress * 0.5)
        explore_weight = 1.0 - exploit_weight
        
        # Combine components with dynamic weights
        s = (exploit_weight * mu_sum + 
             explore_weight * sigma_norm * 2.0 + 
             dist_to_obs * min(stagnant, 3))
        
        scores.append(s)
    
    return scores