def score_pool(context):
    """Exploitation with adaptive uncertainty bonus: sum of means plus UCB-style uncertainty scaled by progress, with early exploitation bias."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early: more exploration (higher weight on uncertainty); later: more exploitation
        ucb_weight = 2.5 * (1.0 - progress ** 0.7)
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores