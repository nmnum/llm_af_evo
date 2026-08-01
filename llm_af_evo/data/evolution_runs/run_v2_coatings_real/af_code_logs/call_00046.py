def score_pool(context):
    """Adaptive exploitation-weighted uncertainty bonus with progress-aware blend and fixed UCB-like ratio."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: more exploitation early, more exploration later
        w_exploit = 0.2 + 0.8 * (1 - progress ** 0.5)
        # Combine with fixed UCB-like ratio to balance mean and uncertainty
        scores.append(w_exploit * mu_sum + 2.0 * sigma_sum)
    return scores