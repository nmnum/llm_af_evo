def score_pool(context):
    """Exploitation with progressive uncertainty weighting: higher predicted means, scaled by progress-driven UCB weight."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        # Scale uncertainty contribution based on progress: less early, more later
        ucb_weight = 0.5 + 0.5 * progress  # 0.5 to 1.0
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores