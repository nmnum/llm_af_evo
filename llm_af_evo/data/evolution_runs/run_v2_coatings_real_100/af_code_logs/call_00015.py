def score_pool(context):
    """Score by predicted objective sum plus a progress-scaled uncertainty bonus, with early exploitation bias and a novelty term based on distance to nearest observed point."""
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
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            novelty_bonus = np.min(distances)
        else:
            novelty_bonus = 1.0
        scores.append(mu_sum + ucb_weight * sigma_sum + 0.5 * novelty_bonus)
    return scores