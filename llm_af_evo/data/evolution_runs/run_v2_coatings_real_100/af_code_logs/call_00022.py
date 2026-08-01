def score_pool(context):
    """Exploitation with dynamic uncertainty bonus: sum of means plus a UCB-style uncertainty term that scales with progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Dynamic uncertainty weight: higher early, lower late
        w = 2.0 * (1.0 - progress)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + w * sigma_norm)
    return scores