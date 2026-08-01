def score_pool(context):
    """Exploitation with progressive uncertainty weighting: balance mean prediction and uncertainty based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Progressive weighting: early = more uncertainty (beta), late = more exploitation (1-beta)
        beta = 0.5 + 0.5 * progress  # starts at 0.5, increases to 1.0
        scores.append(mu_sum + beta * sigma_norm)
    return scores