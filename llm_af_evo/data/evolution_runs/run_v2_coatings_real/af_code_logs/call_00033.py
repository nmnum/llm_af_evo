def score_pool(context):
    """Adaptive exploitation-weighted uncertainty bonus: blend of predicted mean and uncertainty, tuned by progress with a fixed UCB-like bonus."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: early = more explore, late = more exploit, with a fixed UCB bonus
        w_exploit = 0.3 + 0.7 * (1 - progress)
        scores.append(w_exploit * mu_sum + 2.0 * sigma_sum)
    return scores