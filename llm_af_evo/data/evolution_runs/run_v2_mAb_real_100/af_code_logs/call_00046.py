def score_pool(context):
    """Exploitation with dynamic uncertainty weighting and novelty bonus: use higher uncertainty bonus early, reduce it later based on campaign progress, and add a distance-based novelty term to encourage exploration of unvisited regions."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: start with 2.0, decay to 0.5 as progress increases
        w = 2.0 - (1.5 * progress)
        # Compute novelty bonus as inverse of distance to nearest observed point
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            novelty = 1.0 / (np.min(distances) + 1e-8)  # Add small epsilon to avoid division by zero
        else:
            novelty = 1.0
        scores.append(mu_sum + w * sigma_sum + 0.1 * novelty)
    return scores