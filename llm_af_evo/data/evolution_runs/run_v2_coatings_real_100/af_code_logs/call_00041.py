def score_pool(context):
    """Adaptive UCB with exploitation bias and progressive uncertainty scaling."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Start with strong exploration, decay to balanced weighting
        ucb_weight = 2.5 * (1.0 - progress ** 0.7) + 0.3
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores