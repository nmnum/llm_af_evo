def score_pool(context):
    """Exploitation with dynamic uncertainty bonus: weighted sum of means and normalized uncertainty, adjusted by campaign progress to balance exploration and exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Dynamic weight: more exploitation early, more exploration later
        w_exploit = 1.0 - progress
        w_explore = progress
        scores.append(w_exploit * mu_sum + w_explore * sigma_norm)
    return scores