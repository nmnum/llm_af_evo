def score_pool(context):
    """Exploitation with progressive uncertainty weighting: start greedy, then add exploration as campaign progresses."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Progressive weighting: early = exploitation, late = exploration
        weight_exploitation = max(0.5, 1.0 - progress * 0.5)
        scores.append(weight_exploitation * mu_sum + (1 - weight_exploitation) * sigma_sum)
    return scores