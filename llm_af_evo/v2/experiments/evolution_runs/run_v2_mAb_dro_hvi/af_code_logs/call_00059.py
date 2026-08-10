def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted objective sum,
    but reduce scores for candidates that are too similar to already-selected ones.
    This encourages exploration of diverse regions rather than just high-value points.
    """
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base exploitation score (sum of means)
    mu_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_score = sum(gp[name]["mean"] for name in names)
        mu_scores.append(mu_score)

    if len(context["X_obs"]) == 0:
        return mu_scores

    # Compute novelty penalty based on distance to nearest observed point
    X_pool = np.array([cand['x'] for cand in context["pool"]])
    distances = cdist(X_pool, context["X_obs"])
    min_distances = np.min(distances, axis=1)
    
    # Normalize by the range of each objective dimension (for consistent scaling)  
    x_ranges = [front_range[name] for name in names]
    normalized_min_dists = min_distances / max(x_ranges)

    # Apply a novelty penalty that reduces score for candidates near observed points
    novelty_penalty_factor = 0.5 
    penalties = novelty_penalty_factor * (1 - np.exp(-normalized_min_dists))
    
    return [mu + pen for mu, pen in zip(mu_scores, penalties)]