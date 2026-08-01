def score_pool(context):
    """Exploitation with adaptive uncertainty weighting: balance mean prediction and uncertainty based on campaign progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: start with more exploitation, shift towards uncertainty as progress increases
        weight = 0.7 + 0.3 * progress  # 0.7 to 1.0
        scores.append(weight * mu_sum + (1 - weight) * sigma_sum)
    return scores