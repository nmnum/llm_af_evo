def score_pool(context):
    """Exploitation-weighted uncertainty bonus with adaptive blend and novelty penalty."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: early = more uncertainty, late = more exploitation
        w_exploit = 0.3 + 0.7 * (1 - progress)
        # Add novelty penalty based on distance to nearest observed point
        x = cand["x"]
        if len(context["X_obs"]) > 0:
            dist = np.min(np.linalg.norm(context["X_obs"] - x, axis=1))
            novelty_penalty = 0.05 * dist
        else:
            novelty_penalty = 0.0
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum - novelty_penalty)
    return scores