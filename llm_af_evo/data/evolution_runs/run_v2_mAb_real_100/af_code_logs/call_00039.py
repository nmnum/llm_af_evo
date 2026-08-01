def score_pool(context):
    """Exploitation with dynamic uncertainty weighting and novelty bonus: use higher uncertainty bonus early, reduce it later based on campaign progress, and add a distance-based novelty term."""
    names = context["objective_names"]
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: start with 2.0, decay to 0.5 as progress increases
        w = 2.0 - (1.5 * progress)
        # Compute novelty bonus as inverse of distance to nearest observed point
        if X_obs.size == 0:
            novelty = 0.0
        else:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            min_distance = np.min(distances)
            novelty = 1.0 / (1e-8 + min_distance)  # Avoid division by zero
        scores.append(mu_sum + w * sigma_sum + 0.5 * novelty)
    return scores