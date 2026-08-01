def score_pool(context):
    """Exploitation-weighted by progress, with uncertainty bonus scaled by stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant = context["campaign"]["stagnant_batches"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        # Weight exploitation vs uncertainty based on progress
        exploit_weight = 1.0 - progress * 0.5
        ucb_score = mu_sum * exploit_weight + (1.0 - exploit_weight) * sigma_norm * 3.0
        # Boost score if we're stagnant to encourage exploration
        if stagnant > 0:
            ucb_score *= (1.0 + stagnant * 0.1)
        scores.append(ucb_score)
    return scores