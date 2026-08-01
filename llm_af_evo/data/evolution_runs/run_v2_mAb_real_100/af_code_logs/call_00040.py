def score_pool(context):
    """Exploitation with dynamic uncertainty weighting: use higher uncertainty bonus early, reduce it later based on campaign progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic weight: start with 2.0, decay to 0.5 as progress increases
        w = 2.0 - (1.5 * progress)
        scores.append(mu_sum + w * sigma_sum)
    return scores