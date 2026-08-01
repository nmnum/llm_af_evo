def score_pool(context):
    """Score by predicted objective sum plus an adaptive uncertainty bonus based on progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptive weight: more exploration early, more exploitation later
        w = 1.0 + (5.0 * progress)  # increases from 1.0 to 6.0
        scores.append(mu_sum + w * sigma_sum)
    return scores