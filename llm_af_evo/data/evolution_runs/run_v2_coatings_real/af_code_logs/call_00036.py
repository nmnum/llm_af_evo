def score_pool(context):
    """Exploitation-plus-uncertainty with progress-aware weighting: higher progress = more exploitation, less uncertainty bonus."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: early progress = more uncertainty bonus, later = more exploitation
        w_exploit = 0.3 + 0.7 * progress  # from 0.3 to 1.0
        w_uncert = 1.0 - w_exploit
        scores.append(w_exploit * mu_sum + w_uncert * 2.0 * sigma_sum)
    return scores