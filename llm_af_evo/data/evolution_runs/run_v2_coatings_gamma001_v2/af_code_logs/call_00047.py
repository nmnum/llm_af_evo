def score_pool(context):
    """Adaptive exploitation and uncertainty tradeoff: weighted sum of means and normalized uncertainty, with progress-dependent weights."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: more exploitation early, more exploration later
        w_exploit = 0.7 + 0.3 * (1 - progress)
        w_uncertainty = 1.0 - w_exploit
        scores.append(w_exploit * mu_sum + w_uncertainty * 2.0 * sigma_norm)
    return scores