def score_pool(context):
    """Exploitation-plus-uncertainty with progress-adaptive weights, UCB-style."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: more exploitation early, more exploration later
        w_exploit = 1.0 - progress
        w_explore = progress
        scores.append(w_exploit * mu_sum + w_explore * sigma_sum)
    return scores