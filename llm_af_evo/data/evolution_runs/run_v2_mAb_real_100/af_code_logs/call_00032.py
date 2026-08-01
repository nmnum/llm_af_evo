def score_pool(context):
    """Exploitation with adaptive uncertainty bonus: use UCB-style tradeoff, but shift weight towards exploitation as campaign progresses and add a novelty term to avoid overfitting."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early on, favour uncertainty; later, exploit more
        w_exploit = 0.3 + 0.7 * progress
        ucb_score = w_exploit * mu_sum + (1 - w_exploit) * sigma_sum
        
        # Add novelty bonus: candidates farther from observed points get higher scores
        if len(X_obs) > 0:
            dist_to_observed = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            novelty_bonus = 1.0 / (1.0 + dist_to_observed)
            ucb_score += 0.1 * novelty_bonus
        
        scores.append(ucb_score)
    return scores