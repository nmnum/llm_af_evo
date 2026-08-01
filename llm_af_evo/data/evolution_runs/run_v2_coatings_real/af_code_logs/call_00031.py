def score_pool(context):
    """Exploitation with adaptive uncertainty bonus: UCB-style score that balances mean and std, decreasing uncertainty weight as training progresses."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adaptively balance exploitation and exploration based on progress
        ucb_weight = 2.0 * (1.0 - progress ** 0.5)
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores