def score_pool(context):
    """Exploitation with adaptive uncertainty weighting: higher predicted means, blended with normalized uncertainty based on campaign progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    front_range = context["pareto_front_range"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Adaptive weight: start with low uncertainty influence, increase as progress advances
        adaptive_weight = 0.3 + 0.7 * progress
        scores.append(mu_sum + adaptive_weight * sigma_norm)
    return scores