def score_pool(context):
    """
    Exploitation with uncertainty-aware novelty: rank by predicted objective sum,
    but reduce scores for candidates that are too close to already observed points.
    This promotes exploration of novel regions while still favouring high-yield predictions.
    """
    names = context["objective_names"]
    X_obs = context["X_obs"]
    scores = []
    
    # For each candidate, compute its score as the sum of predicted means
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Compute distance to nearest observed point (in feature space) 
        x_cand = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            min_distance = np.min(distances)
            
            # Reduce score for candidates too close to existing observations
            novelty_penalty = max(0.0, 1.0 - min_distance / 2.0) 
        else:
            novelty_penalty = 1.0
            
        scores.append(mu_sum * novelty_penalty)

    return scores