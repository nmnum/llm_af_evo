def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted objective sum,
    but reduce scores for candidates that are too similar to already-observed points.
    This encourages exploration of diverse regions while still favouring high-yield predictions.
    """
    names = context["objective_names"]
    X_obs = context["X_obs"]
    Y_obs = context["Y_obs"] 
    pool = context["pool"]

    # Compute scores as sum of means
    mu_scores = [sum(cand["gp_posterior"][name]["mean"] for name in names) for cand in pool]

    if len(X_obs) == 0:
        return mu_scores

    # For each candidate, compute distance to nearest observed point (in feature space)
    scores = []
    for i, cand in enumerate(pool):
        x_cand = cand["x"]
        
        distances = np.sqrt(np.sum((X_obs - x_cand)**2, axis=1))
        min_distance = np.min(distances)

        # If candidate is very close to an observed point (within 0.5% of feature space range),
        # reduce its score significantly
        if min_distance < 0.005 * np.sqrt(len(x_cand)):  
            penalty_factor = max(0.1, 1 - min_distance / (0.005 * np.sqrt(len(x_cand))))
            scores.append(mu_scores[i] * penalty_factor)
        else:
            scores.append(mu_scores[i])

    return scores