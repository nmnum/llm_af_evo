def score_pool(context):
    """Exploitation with progressive uncertainty weighting and novelty penalty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Progressive weighting: start with exploitation, add uncertainty as we progress
        weight_exploit = 1.0 - min(0.9, progress * 0.8)
        score = weight_exploit * mu_sum + (1.0 - weight_exploit) * sigma_norm
        # Add novelty penalty: candidates near observed points get lower scores
        if len(X_obs) > 0:
            dist_to_obs = np.min(np.linalg.norm(cand["x"] - X_obs, axis=1))
            score -= 0.1 * dist_to_obs
        scores.append(score)
    return scores