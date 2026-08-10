def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted objective sum,
    then penalize candidates that are too close to existing observations in feature space.
    This encourages exploration of under-represented regions while still exploiting high-value predictions.
    """
    names = context["objective_names"]
    scores = []
    
    # Compute base exploitation score (sum of means)
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Add novelty penalty based on distance to nearest observed point
        x_cand = cand["x"]
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_distance = np.min(distances)
            
            # Scale the novelty penalty inversely with distance (closer = more penalized)
            # Use a small constant to avoid division by zero
            novel_penalty = 0.5 / (min_distance + 1e-8) if min_distance > 0 else 2.0
            
        else:
            novel_penalty = 0.0
        
        scores.append(mu_sum - novel_penalty * np.std([gp[name]["mean"] for name in names]))
    
    return scores