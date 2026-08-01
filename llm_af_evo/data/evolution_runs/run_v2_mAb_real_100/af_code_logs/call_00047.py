def score_pool(context):
    """Exploitation with progressive uncertainty weighting: start greedy, then add uncertainty bonus as campaign progresses."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Progressive weighting: early = pure exploitation, late = more uncertainty-aware
        weight = 0.5 + 0.5 * progress  # from 0.5 to 1.0
        scores.append(mu_sum + weight * sigma_sum)
    return scores