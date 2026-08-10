def score_pool(context):
    """Exploitation with novelty bonus: rank by predicted objective sum, then boost candidates far from existing observations."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        
        # Compute distance to nearest observed point
        x_cand = cand["x"]
        if len(context["X_obs"]) == 0:
            novelty_score = 1.0
        else:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_distance = np.min(distances)
            # Invert distance to get novelty score (higher is better), with a small floor
            novelty_score = max(1e-6, 1.0 / (min_distance + 1e-6))
        
        scores.append(mu_sum * (1.0 + 0.5 * novelty_score)) 
    return scores