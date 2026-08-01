def score_pool(context):
    """Exploitation with progress-aware uncertainty bonus: use UCB-style tradeoff, but shift weight towards exploitation as campaign progresses."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Early on, favour uncertainty; later, exploit more
        w_exploit = 0.3 + 0.7 * progress
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores