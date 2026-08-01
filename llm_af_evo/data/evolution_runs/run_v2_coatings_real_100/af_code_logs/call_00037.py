def score_pool(context):
    """Blend exploitation and uncertainty with a progress-aware weight, scaled by front range for normalization."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: early = more uncertainty, late = more exploitation
        w_exploit = 0.3 + 0.7 * (1 - progress)
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores