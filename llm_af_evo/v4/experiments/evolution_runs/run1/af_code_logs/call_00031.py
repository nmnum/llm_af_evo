def score_pool(context):
    """Combine UCB-style exploration credit with explicit novelty term."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] for name in names)
        ucb_score = mu_sum + 1.0 * sigma_norm
        min_dist_to_observed = np.min(np.linalg.norm(context["X_obs"] - cand["x"], axis=1))
        novelty_bonus = 1.0 / (min_dist_to_observed + 1e-8)
        scores.append(ucb_score + 0.5 * novelty_bonus)
    return scores