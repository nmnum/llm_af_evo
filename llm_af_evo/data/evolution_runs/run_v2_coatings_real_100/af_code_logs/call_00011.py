def score_pool(context):
    """Exploitation-plus-uncertainty with progress-aware weighting: higher progress = more exploitation, less uncertainty bonus."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Weight uncertainty bonus inversely to progress (less exploration late)
        weight = 1.0 - progress
        scores.append(mu_sum + weight * sigma_sum)
    return scores