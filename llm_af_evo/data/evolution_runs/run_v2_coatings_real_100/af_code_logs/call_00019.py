def score_pool(context):
    """Score by predicted objective sum plus a UCB-style uncertainty bonus weighted by progress."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Weight uncertainty bonus by progress: more early exploration, less later
        weight = 2.0 * (1 - progress)
        scores.append(mu_sum + weight * sigma_sum)
    return scores