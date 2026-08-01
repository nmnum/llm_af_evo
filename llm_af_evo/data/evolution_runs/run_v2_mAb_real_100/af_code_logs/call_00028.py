def score_pool(context):
    """Exploitation with uncertainty-aware diversity: sum of means plus a novelty bonus based on distance to nearest observed point."""
    names = context["objective_names"]
    scores = []
    X_obs = context["X_obs"]
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Compute novelty bonus as inverse of distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) == 0:
            novelty_bonus = 0.0
        else:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            # Avoid division by zero; set a small minimum distance if needed
            novelty_bonus = 1.0 / (min_distance + 1e-8)
        scores.append(mu_sum + 0.5 * novelty_bonus)
    return scores