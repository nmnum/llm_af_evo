def score_pool(context):
    """Hybrid of exploitation and novelty with progress-aware weighting."""
    X_obs = context["X_obs"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = gp["Tm"]["mean"] + gp["kD"]["mean"] + gp["viscosity"]["mean"]
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = dists.min()
        exploit_weight = 1.0 - progress
        scores.append(exploit_weight * mu_sum + (1.0 - exploit_weight) * novelty)
    return scores