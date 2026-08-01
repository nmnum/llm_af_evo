def score_pool(context):
    """Exploitation with progressive uncertainty weighting: early exploration, late exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Progressive weighting: start with high uncertainty weight (exploration) and decay to low (exploitation)
        beta = 2.0 * (1.0 - progress)
        scores.append(mu_sum + beta * sigma_norm)
    return scores