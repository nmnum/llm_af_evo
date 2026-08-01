def score_pool(context):
    """Exploitation with uncertainty balance: sum of means plus a progress-adapted uncertainty bonus."""
    names = context["objective_names"]
    scores = []
    progress = context["campaign"]["progress"]
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Adapt uncertainty weight based on progress: more exploration early, less later
        ucb_weight = 1.0 - progress  # decreases from 1.0 to 0.0
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores