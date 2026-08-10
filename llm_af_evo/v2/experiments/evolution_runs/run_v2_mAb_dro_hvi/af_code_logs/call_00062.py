def score_pool(context):
    """
    Exploitation with uncertainty-aware diversity: rank by predicted objective sum,
    then reduce scores for candidates that are too similar (in feature space) 
    to already-selected top performers, encouraging exploration of diverse regions.
    """
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Get all observed points and their corresponding predictions
    X_obs = context["X_obs"]  
    Y_obs = context["Y_obs"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Base score: sum of predicted means (exploitation)
        mu_sum = sum(gp[name]["mean"] for name in names)

        # Diversity penalty based on proximity to observed points
        x_cand = cand["x"]
        min_dist_to_observed = float('inf')
        if len(X_obs) > 0:
            distances_squared = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist_to_observed = np.sqrt(np.min(distances_squared))
        
        # Apply inverse distance penalty (higher score for candidates farther from observed points)
        diversity_score = max(0.5, 1 / (min_dist_to_observed + 1e-8)) 

        scores.append(mu_sum * diversity_score)

    return scores