def score_pool(context):
    """Score by predicted objective sum plus a progress-scaled uncertainty bonus, with early exploitation bias and novelty reward."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early: more exploration (higher weight on uncertainty); later: more exploitation
        ucb_weight = 3.0 * (1.0 - progress ** 0.5)
        # Add novelty bonus based on distance to nearest observed point
        distances = np.sum((cand["x"] - context["X_obs"]) ** 2, axis=1)
        novelty_bonus = np.exp(-np.min(distances) / (2 * np.var(context["X_obs"], axis=0).sum()))
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty_bonus)
    return scores