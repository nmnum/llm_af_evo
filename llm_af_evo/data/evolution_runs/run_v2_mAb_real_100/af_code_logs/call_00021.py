def score_pool(context):
    """Exploitation with progressive uncertainty weighting and novelty bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    X_obs = context["X_obs"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Progressive weighting: start with exploitation, add uncertainty as we progress
        w_exploit = 1.0 - min(progress * 2.0, 1.0)
        w_uncertainty = min(progress * 2.0, 1.0)
        base_score = w_exploit * mu_sum + w_uncertainty * sigma_norm
        # Add novelty bonus: candidates farther from observed points get higher scores
        if len(X_obs) > 0:
            dist_to_obs = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            novelty_bonus = dist_to_obs / (np.linalg.norm(front_range) + 1e-8)
            base_score += 0.1 * novelty_bonus
        scores.append(base_score)
    return scores