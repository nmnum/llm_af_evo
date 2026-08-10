def score_pool(context):
    """Exploitation with novelty penalty: sum of means minus a scaled distance-to-nearest-observed-point term."""
    names = context["objective_names"]
    scores = []
    X_obs = context["X_obs"]
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Compute distance to nearest observed point
        x_cand = cand["x"]
        if len(X_obs) == 0:
            dist_to_nearest = float('inf')
        else:
            distances = np.sqrt(np.sum((X_obs - x_cand)**2, axis=1))
            dist_to_nearest = np.min(distances)
        
        # Novelty penalty: higher distance => more preferred
        novelty_score = 1.0 / (1e-8 + dist_to_nearest) 
        scores.append(mu_sum + 0.5 * novelty_score)
    return scores