def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted mean sum,
    but penalize candidates that are too close (in feature space) to already-
    observed points or existing Pareto front members, encouraging exploration.
    """
    names = context["objective_names"]
    scores = []
    
    # Compute distances from each candidate to all previously seen points
    X_obs = np.vstack([context["X_obs"], context["pareto_front"]]) if len(context["pareto_front"]) > 0 else context["X_obs"]

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Base exploitation score (higher is better)
        mu_sum = sum(gp[name]["mean"] for name in names)

        # Compute minimum distance to any observed point
        dist_to_observed = np.min(np.sum((X_obs - cand["x"])**2, axis=1)) if len(X_obs) > 0 else float('inf')

        # Convert squared Euclidean distances into a novelty penalty (smaller is better)
        epsilon = 1e-8
        novel_score = mu_sum / (dist_to_observed + epsilon)

        scores.append(novel_score)
        
    return scores