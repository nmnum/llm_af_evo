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
        # Progressive weighting: start with 1.0 exploitation, reduce to 0.5 as progress increases
        w_exploit = 1.0 - 0.5 * progress
        # Add uncertainty term with progressive weight
        ucb_score = w_exploit * mu_sum + (1.0 - w_exploit) * sigma_norm
        # Add novelty penalty: candidates closer to observed points get lower scores
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand["x"], axis=1)
            min_distance = np.min(distances)
            # Invert distance to reward distant candidates (novelty)
            novelty_penalty = 1.0 / (1.0 + min_distance)
            ucb_score *= novelty_penalty
        scores.append(ucb_score)
    return scores