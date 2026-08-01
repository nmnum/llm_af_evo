def score_pool(context):
    """Exploitation with dynamic uncertainty bonus; weight mu_sum and sigma_sum dynamically based on campaign progress."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: more exploitation early, more uncertainty bonus later
        w_exploit = max(0.7, 1.0 - progress * 0.3)
        w_uncertainty = 1.0 - w_exploit
        scores.append(w_exploit * mu_sum + w_uncertainty * sigma_sum * 2.0)
    return scores