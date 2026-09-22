def score_pool(context):
    """Adapts exploration-exploitation balance using progress and uncertainty scaling while incorporating historical diversity via novel candidate penalization."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic exploitation weight that increases with progress
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5*progress))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        # Combine exploitation and uncertainty with dynamic weighting
        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_norm
        
        # Penalize candidates that are too similar to already observed ones 
        x_cand = cand["x"]
        
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(x_cand[None, :] - context["X_obs"], axis=1)
            min_distance = np.min(distances) 
            
            # Apply a penalty for candidates too close to existing observations
            threshold = max(front_range.values()) * 0.05  
            
            if min_distance < threshold:
                score *= (1.0 - 0.3 * (threshold - min_distance)/threshold)
                
        scores.append(score)

    return scores