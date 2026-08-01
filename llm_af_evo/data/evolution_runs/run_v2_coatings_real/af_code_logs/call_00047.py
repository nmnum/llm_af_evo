def score_pool(context):
    """Exploitation with adaptive uncertainty bonus: sum of means plus UCB-style uncertainty weighted by progress, scaled by inverse of observed range."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_weight = 1.0 + 2.0 * progress
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores