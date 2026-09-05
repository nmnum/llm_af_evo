def score_pool(context):
    """Greedy proximity-based suppression of near-duplicates in feature space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Normalize candidates' features to [0, 1] for distance calculation  
    pool_x_normed = np.array([cand['x'] / (np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)) 
                              if len(context["X_obs"]) > 0 else cand['x']
                              for cand in context["pool"]])
    
    # Normalize previously observed points
    X_normed = None  
    if len(context["X_obs"]) > 0:
        x_range = np.max(context["X_obs"], axis=0) - np.min(context["X_obs"], axis=0)
        mask = (x_range != 0)
        X_normed = context["X_obs"].copy()
        X_normed[:,mask] /= x_range[mask]
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        # Exploitation signal: mean of objectives
        mu_sum = sum(gp[name]["mean"] for name in names)
                
        # Uncertainty term (standard deviation) 
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        score = 1.0 * mu_sum + 2.5 * sigma_norm

        if len(context["X_obs"]) > 0:
            cand_x = pool_x_normed[i]
            
            # Suppress candidates that are too close to already observed points
            distances = np.linalg.norm(X_normed - cand_x, axis=1)
            min_distance = np.min(distances) 
                
            # If candidate is very near an existing point (within 0.5% of feature space), suppress score significantly  
            if min_distance < 0.005:
                score *= 0.2
                
        scores.append(score)

    return scores