def score_pool(context):
    """Score by predicted objective sum plus uncertainty bonus scaled by progress and adjusted for stagnation, UCB-style."""
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_sum = sum(gp[name]["std"] for name in names)
        # Dynamic UCB weight: decrease exploration as progress increases, and reduce further if stagnant
        ucb_weight = 2.0 * max(0.1, (1.0 - progress) * (1.0 - 0.3 * min(stagnant_batches, 5) / 5.0))
        scores.append(mu_sum + ucb_weight * sigma_sum)
    return scores