def score_pool(context):
    """Score by predicted objective sum plus a progress-scaled uncertainty bonus, with early exploitation bias and a novelty term."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early: more exploration (higher weight on uncertainty); later: more exploitation
        ucb_weight = 3.0 * (1.0 - progress ** 0.5)
        # Add novelty term based on distance to nearest observed point
        distances = np.sum((context["X_obs"] - cand["x"]) ** 2, axis=1)
        novelty = 1.0 / (np.min(distances) + 1e-8)
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty)
    return scores