def score_pool(context):
    """Exploitation with dynamic uncertainty weighting and novelty bonus: use higher uncertainty bonus early, reduce it later based on campaign progress, and reward candidates far from previously observed points."""
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
        # Compute distance to nearest observed point
        if X_obs.size > 0:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            novelty = np.min(distances)
        else:
            novelty = 1.0  # If no observations, assume maximum novelty
        # Combine exploitation and uncertainty with novelty bonus
        scores.append(mu_sum + w * sigma_sum + 0.1 * novelty)
    return scores