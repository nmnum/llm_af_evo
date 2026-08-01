def score_pool(context):
    """Adaptive UCB-style score: blend of mean and uncertainty with progress-aware weights, plus novelty penalty."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: more exploitation early, more exploration late
        w_exploit = 0.2 + 0.8 * (1 - progress)
        ucb_score = w_exploit * mu_sum + (1 - w_exploit) * sigma_sum
        # Add novelty bonus: candidates farther from observed points get higher scores
        if len(X_obs) > 0:
            dist_to_obs = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            novelty_bonus = 0.5 * dist_to_obs
        else:
            novelty_bonus = 0.0
        scores.append(ucb_score + novelty_bonus)
    return scores