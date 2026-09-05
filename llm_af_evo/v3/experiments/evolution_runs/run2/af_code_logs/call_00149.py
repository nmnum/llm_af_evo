def score_pool(context):
    """Score by predicted objective sum plus a fixed-weight uncertainty bonus, UCB-style."""
    names = context["objective_names"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        scores.append(mu_sum + 2.0 * sigma_sum)
    return scores