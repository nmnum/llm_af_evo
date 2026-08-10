def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted mean sum,
    but reduce scores for candidates that are too similar to already-observed points.
    This encourages exploration of novel regions while still favouring high-value predictions.
    """
    names = context["objective_names"]
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"] 
    pool = context["pool"]

    # Compute scores as mean sum
    mu_scores = [sum(cand["gp_posterior"][name]["mean"] for name in names) for cand in pool]

    if len(X_obs) == 0:
        return mu_scores

    # For each candidate, compute min distance to observed points (normalized by front range)
    scores = []
    front_range = context["pareto_front_range"]
    
    for i, cand in enumerate(pool):
        x_cand = cand["x"] 
        distances = np.sqrt(np.sum(((X_obs - x_cand) / list(front_range.values()))**2, axis=1))
        min_distance = np.min(distances)
        
        # Reduce score based on inverse of distance (novelty penalty)
        novelty_factor = 1.0 + 5.0 * np.exp(-min_distance ** 2)

        scores.append(mu_scores[i] / novelty_factor)

    return scores