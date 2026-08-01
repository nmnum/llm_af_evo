def score_pool(context):
    """Exploitation with adaptive uncertainty bonus: sum of means plus UCB-style uncertainty scaled by progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: start with low uncertainty bonus, increase as progress increases
        ucb_weight = 0.5 * progress
        scores.append(mu_sum + ucb_weight * sigma_norm)
    return scores