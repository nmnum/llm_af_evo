def score_pool(context):
    """Combine UCB-style exploration credit with explicit novelty term."""
    names = context["objective_names"]
    scores = []
    X_obs = context["X_obs"]
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # UCB-style exploration: sum of means plus uncertainty bonus
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] for name in names)  # no normalisation needed here
        
        ucb_score = mu_sum + 1.0 * sigma_norm

        # Novelty term: distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) == 0:
            novelty = 1.0  # fallback when nothing has been observed yet
        else:
            distances = np.sqrt(np.sum((X_obs - cand_x)**2, axis=1))
            min_distance = np.min(distances)
            novelty = 1.0 / (min_distance + 1e-8)  # invert distance to make it a score

        scores.append(ucb_score + 0.5 * novelty)

    return scores