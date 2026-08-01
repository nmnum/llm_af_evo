def score_pool(context):
    """Exploitation with uncertainty bonus: sum of means plus a scaled uncertainty term, adjusted by progress to balance exploration and exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Scale uncertainty bonus based on progress: more exploration early, less later
        uncertainty_weight = 0.5 * (1 - progress)
        scores.append(mu_sum + uncertainty_weight * sigma_norm)
    return scores