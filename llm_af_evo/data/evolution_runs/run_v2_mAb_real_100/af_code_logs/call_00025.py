def score_pool(context):
    """Exploitation with progressive uncertainty weighting and novelty bonus: higher predicted means, scaled by progress-driven UCB weight, plus distance from observed points."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Scale uncertainty contribution based on progress: less early, more later
        ucb_weight = 0.5 + 0.5 * progress  # 0.5 to 1.0
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Add novelty bonus based on distance from observed points
        if len(X_obs) > 0:
            dist = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            novelty_bonus = 1.0 / (1.0 + dist)  # invert and normalize
        else:
            novelty_bonus = 1.0
        scores.append(mu_sum + ucb_weight * sigma_sum + novelty_bonus)
    return scores