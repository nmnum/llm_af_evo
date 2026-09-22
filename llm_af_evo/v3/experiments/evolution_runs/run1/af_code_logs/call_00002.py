def score_pool(context):
    """Combine UCB-style exploration with novelty term based on distance to nearest observed point."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] for name in names)
        ucb_score = mu_sum + 0.5 * sigma_norm
        # Novelty term: inverse of distance to nearest observed point
        x_cand = cand["x"]
        if len(context["X_obs"]) == 0:
            novelty = 1e6  # No observations, so very novel
        else:
            distances = np.linalg.norm(context["X_obs"] - x_cand, axis=1)
            min_distance = np.min(distances)
            novelty = 1. / (min_distance + 1e-9) if min_distance > 0 else 1e6
        scores.append(ucb_score + 0.5 * novelty)
    return scores