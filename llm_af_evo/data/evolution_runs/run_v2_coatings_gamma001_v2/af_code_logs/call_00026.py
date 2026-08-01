def score_pool(context):
    """Exploitation with uncertainty bonus: sum of means plus a scaled uncertainty term, where the scale decreases over time to balance exploration and exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Decrease the uncertainty bonus weight as progress increases
        beta = 2.0 * (1.0 - progress)
        scores.append(mu_sum + beta * sigma_norm)
    return scores