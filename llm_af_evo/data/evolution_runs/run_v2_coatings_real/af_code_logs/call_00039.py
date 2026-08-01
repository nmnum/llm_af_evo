def score_pool(context):
    """Adaptive exploitation-exploration balance with uncertainty bonus, blending UCB and progress-weighted mean strategies."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Start with UCB-style blend, then adjust exploitation weight based on progress
        ucb_weight = 2.0 * (1.0 - progress ** 0.5)  # Slightly slower decay than linear
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores