def score_pool(context):
    """Suppresses candidates that are too similar to already-observed points, favoring diversity in predicted objective space."""
    names = context["objective_names"]
    X_obs = context["X_obs"] 
    Y_obs = context["Y_obs"]
    
    # Compute distances from each candidate to the nearest observed point
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        if len(X_obs) == 0:
            distance_to_nearest = np.inf  
        else: 
            diffs = X_obs - x_cand[None, :]
            distances_sq = np.sum(diffs**2, axis=1)
            distance_to_nearest = np.sqrt(np.min(distances_sq))
        
        # Predicted objectives
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)

        # Suppress candidates that are too close to observed points (distance < threshold)  
        if distance_to_nearest < 0.1:
            scores.append(-np.inf)
        else: 
            scores.append(mu_sum)
    
    return scores