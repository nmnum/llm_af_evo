def score_pool(context):
    """Progressive exploitation with uncertainty bonus: blend of UCB-style exploration and adaptive weighting based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Start with higher exploration weight early, reduce as progress increases
        beta = 1.5 * (1.0 - progress ** 2)
        scores.append(mu_sum + beta * sigma_norm)
    return scores