def score_pool(context):
    """Exploitation with progress-aware uncertainty bonus and novelty penalty: UCB-style score with dynamic weight and distance from observed points."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic UCB weight that decreases with progress
        ucb_weight = 2.0 * (1.0 - progress)
        # Add novelty bonus: score inversely proportional to distance to nearest observed point
        if len(X_obs) > 0:
            distances = np.sqrt(np.sum((X_obs - cand["x"])**2, axis=1))
            novelty_bonus = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty_bonus = 1.0
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty_bonus)
    return scores