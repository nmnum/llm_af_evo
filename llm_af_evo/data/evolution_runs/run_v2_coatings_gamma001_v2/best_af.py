def score_pool(context):
    """Exploitation with uncertainty bonus: sum of means plus a scaled uncertainty term to encourage exploration near the Pareto front."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        scores.append(mu_sum + 0.5 * sigma_norm)
    return scores