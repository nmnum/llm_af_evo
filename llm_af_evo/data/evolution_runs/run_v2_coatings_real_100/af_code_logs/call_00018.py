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
        # Early: more exploration (higher weight on uncertainty); later: more exploitation
        ucb_weight = 3.0 * (1.0 - progress ** 0.5)
        score = mu_sum + ucb_weight * sigma_sum
        # Add a novelty penalty based on distance to nearest observed point
        if len(X_obs) > 0:
            dist_to_obs = np.min(np.linalg.norm(X_obs - cand["x"], axis=1))
            novelty_penalty = 0.1 * dist_to_obs
            score -= novelty_penalty
        scores.append(score)
    return scores