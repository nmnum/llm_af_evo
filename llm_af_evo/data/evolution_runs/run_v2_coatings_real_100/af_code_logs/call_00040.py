def score_pool(context):
    """Exploitation with progress-aware uncertainty bonus: sum of means plus UCB-style uncertainty scaled by campaign progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Scale uncertainty bonus by progress: less exploration early, more later
        ucb_weight = 2.0 * (1.0 - progress)
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores