def score_pool(context):
    """Adaptive exploitation-weighted uncertainty bonus with progress-aware blending and fixed UCB-like bonus."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: more exploitation as progress increases, but never fully exploit
        w_exploit = 0.2 + 0.6 * (1 - progress**0.5)
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores