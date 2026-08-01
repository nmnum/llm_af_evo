def score_pool(context):
    """Score by predicted objective sum plus a progress-scaled uncertainty bonus, with early exploitation bias and novelty reward."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early: more exploration (higher weight on uncertainty); later: more exploitation
        ucb_weight = 3.0 * (1.0 - progress ** 0.5)
        # Add novelty bonus based on distance to nearest observed point
        if len(X_obs) > 0:
            distances = np.sum((X_obs - cand["x"]) ** 2, axis=1)
            novelty_bonus = np.exp(-np.min(distances) / (2 * 0.1 ** 2))  # Adjust 0.1 for novelty sensitivity
        else:
            novelty_bonus = 1.0
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty_bonus)
    return scores