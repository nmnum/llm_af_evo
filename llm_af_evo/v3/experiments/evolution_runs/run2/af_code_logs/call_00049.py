def score_pool(context):
    """Suppresses candidates that are too close to already-observed points, favoring distant exploration."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    
    # Compute squared distances from each candidate in pool to the nearest observed point
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"]
        
        if len(X_obs) == 0:
            min_dist_sq = np.inf  
        else:
            dists_sq = np.sum((X_obs - x_cand)**2, axis=1)
            min_dist_sq = np.min(dists_sq)

        # Normalize distance by the range of features
        feature_range = np.max(X_obs, axis=0) - np.min(X_obs, axis=0)
        if not np.any(feature_range):
            normalized_distance = 0.0 
        else:
            normalized_distance = min_dist_sq / np.sum(feature_range**2)

        # Prefer candidates that are further from existing observations
        score = max(1e-8, (normalized_distance)**(-3)) 

        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        scores.append(mu_sum * score)  # combine exploitation with novelty bonus

    return scores