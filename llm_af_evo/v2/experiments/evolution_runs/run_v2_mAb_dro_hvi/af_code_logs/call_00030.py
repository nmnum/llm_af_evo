def score_pool(context):
    """Exploitation with novelty penalty: sum of means minus a scaled uncertainty term, plus distance-based diversity reward."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Compute base exploitation score (mean sum)
    cand_scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Use a decaying uncertainty weight based on campaign progress
        w_uncertainty = 0.5 * (1 - context["campaign"]["progress"])
        cand_scores.append(mu_sum - w_uncertainty * sigma_norm)

    # Add novelty bonus: reward candidates far from existing observations
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        for i, score in enumerate(cand_scores):
            x_cand = context["pool"][i]["x"]
            distances = np.linalg.norm(X_obs - x_cand, axis=1)
            min_distance = np.min(distances)
            # Reward candidates that are farther from observed points
            novelty_bonus = 2.0 * (min_distance / max(1e-6, np.std(X_obs))) if len(X_obs) > 0 else 0.
            cand_scores[i] += novelty_bonus

    return cand_scores