def score_pool(context):
    """Score by predicted objective sum plus a progress-scaled uncertainty bonus, with early exploitation bias and novelty penalty."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early: more exploration; later: more exploitation
        ucb_weight = 3.0 * (1.0 - progress ** 0.5)
        score = mu_sum + ucb_weight * sigma_sum
        # Add novelty penalty: candidates closer to observed points get lower scores
        if len(X_obs) > 0:
            distances = np.sqrt(np.sum((X_obs - cand["x"]) ** 2, axis=1))
            min_distance = np.min(distances)
            # Penalty increases as distance decreases (max penalty at 0)
            score -= 0.5 * min_distance
        scores.append(score)
    return scores