def score_pool(context):
    """Adaptive exploitation-exploration balance with uncertainty bonus, improving on fixed UCB and progress-weighted blends."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: more exploitation as progress increases, but with a floor to maintain some exploration
        w_exploit = 0.5 + 0.5 * (1 - progress ** 0.5)  # Slower decay than linear
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores