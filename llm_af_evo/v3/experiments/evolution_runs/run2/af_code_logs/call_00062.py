def score_pool(context):
    """Suppresses candidates that are too close to already observed points by penalizing their acquisition scores."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Use a smooth transition from early exploration (UCB-like) to later exploitation
    w_exploit = 1.0 / (1.0 + np.exp(-24 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized mean sum for exploitation signal  
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
                
        # Uncertainty term (normalized standard deviation) 
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)

        score = w_exploit * mu_sum_norm + (1.0 - w_exploit) * sigma_norm
        
        # Apply proximity-based suppression: if candidate is too close to any observed point, reduce its score
        x_cand = cand["x"]
        min_dist_to_observed = float('inf')
        
        for x_obs in context["X_obs"]:
            dist = np.linalg.norm(x_cand - x_obs)
            min_dist_to_observed = min(min_dist_to_observed, dist)

        # If the candidate is very close to an observed point (e.g., within 1% of feature space), suppress its score
        if min_dist_to_observed < 0.01 * np.sqrt(len(x_cand)):
            penalty_factor = max(0.05, 2.0 - progress) # Stronger suppression early on  
            score *= penalty_factor

        scores.append(score)
    
    return scores