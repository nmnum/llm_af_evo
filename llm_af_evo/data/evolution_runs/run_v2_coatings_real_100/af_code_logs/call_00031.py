def score_pool(context):
    """Blend exploitation and uncertainty with a dynamic weight that shifts from exploration to exploitation."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: start with exploration, shift to exploitation
        w_exploit = 0.3 + 0.7 * (1 - progress ** 0.5)
        scores.append(w_exploit * mu_sum + (1 - w_exploit) * sigma_sum)
    return scores